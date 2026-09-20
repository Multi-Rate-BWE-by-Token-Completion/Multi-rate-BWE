"""Multi-rate bandwidth extension by token completion: predictor training.

One script covers every predictor in the paper. Three config switches:

    codec         "spectrostream" (spectral-domain) or "dac" (waveform-domain)
    cutoff_rates  list of input rates drawn per batch, in Hz. Four rates gives
                  the multi-rate predictor; a single rate gives a per-rate one.
    rate_emb      false: the cutoff is never supplied and must be inferred from
                  the codes (the model proposed in the paper).
                  true: a learned embedding of the cutoff index is added to the
                  pooled input (the "+ rate emb." ablation).

The codec is frozen: only the transformer is trained. Training is teacher-forced
and autoregressive over RVQ depth, with a cross-entropy loss summed over steps.

The cutoff draw is seeded by the trainer step so that all DDP ranks agree on the
rate for a given batch; ranks drawing different cutoffs would mix bandwidths
within one gradient step.

Structure and much of the plumbing are inherited from Descript's Audio Codec
(https://github.com/descriptinc/descript-audio-codec) and audiotools; see NOTICE.
"""
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from einops import rearrange
from audiotools import AudioSignal
from audiotools import ml
from audiotools import STFTParams
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset as BaseAudioDataset
from audiotools.data.datasets import AudioLoader as BaseAudioLoader
from audiotools.ml.decorators import Tracker, timer, when
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import multiclass_accuracy

sys.path.append(os.getcwd())
import sps  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))

Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)
AdamW = argbind.bind(torch.optim.AdamW, "transformer")
SpS = argbind.bind(sps.model.SpS)
TransformerModel = argbind.bind(sps.model.TransformerModel)
RateTransformerModel = argbind.bind(sps.model.RateTransformerModel)

# --------------------------------------------------------------------------
# Dataset plumbing, mirroring train_codec.py so the frozen encoder sees exactly
# the data distribution and STFT parameters it was trained on.
# --------------------------------------------------------------------------
AudioDataset = argbind.bind(BaseAudioDataset, "train", "val")
AudioLoader = argbind.bind(BaseAudioLoader, "train", "val")

_tfm_filter = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform", "Compose", "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=_tfm_filter)

TARGET_STFT_PARAMS = STFTParams(window_length=960, hop_length=480)


def compute_target_stft(signal):
    signal.stft_params = TARGET_STFT_PARAMS
    signal.stft()
    return signal


class STFTAudioDataset(BaseAudioDataset):
    @staticmethod
    def collate(list_of_dicts, n_splits=None):
        batch = BaseAudioDataset.collate(list_of_dicts, n_splits=n_splits)

        def _apply(output):
            if isinstance(output, dict):
                if "signal" in output:
                    output["signal"] = compute_target_stft(output["signal"])
                else:
                    for v in output.values():
                        _apply(v)

        _apply(batch)
        return batch


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    return transforms.Compose(
        transforms.Compose(*to_tfm(preprocess), name="preprocess"),
        transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob),
        transforms.Compose(*to_tfm(postprocess), name="postprocess"),
    )


@argbind.bind("train", "val", "test")
def build_dataset(sample_rate: int = 48000, folders: dict = None, weights: list = None):
    # folders entries may be directories or .csv file lists; weights set the
    # per-source draw probability (requires AudioDataset.without_replacement=false).
    if weights is not None:
        total = float(sum(weights))
        weights = [w / total for w in weights]
    loader = AudioLoader(sources=folders["music_hq"], weights=weights)
    transform = build_transform()
    dataset = STFTAudioDataset(loader, sample_rate, transform=transform)
    dataset.transform = transform
    return dataset


@argbind.bind("transformer")
def ScheduleLR(optimizer, T_max: int = 100000, eta_min: float = 1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


@argbind.bind(without_prefix=True)
def arch(codec: str = "spectrostream", rate_emb: bool = False):
    """The two architecture switches. `codec` selects the frozen tokenizer;
    `rate_emb` selects whether the cutoff index is given to the transformer."""
    codec = str(codec).lower()
    assert codec in ("spectrostream", "dac"), f"codec must be spectrostream|dac, got {codec!r}"
    return codec, bool(rate_emb)


def new_transformer():
    _, rate_emb = arch()
    return RateTransformerModel(test=None) if rate_emb else TransformerModel(test=None)


def call_transformer(transformer, inp, step, rate_idx, accel=None):
    """Forward pass, passing rate_idx only to a model that has a rate embedding."""
    model = accel.unwrap(transformer) if accel is not None else transformer
    if getattr(model, "rate_emb", None) is not None:
        return transformer(inp, step, rate_idx=rate_idx)
    return transformer(inp, step)


@dataclass
class State:
    generator: object                   # frozen codec (SpS or DAC)
    transformer: object
    optimizer: AdamW
    scheduler: ScheduleLR
    criterion: torch.nn.CrossEntropyLoss
    train_data: object
    val_data: object
    tracker: Tracker


def _spectrogram(signal):
    """AudioSignal -> (B, 2, F, T) real/imag, Nyquist bin dropped, as the codec expects."""
    x = rearrange(torch.view_as_real(signal.stft_data), "b 1 f t c -> b c f t")
    return x[:, :, :-1, :]


def band_limit(signal, cutoff_sr: int):
    """Band-limit by resampling down and back up.

    Done in the time domain rather than by zeroing STFT bins so that the input
    carries the same resampling-filter rolloff a genuinely low-sample-rate file
    would have -- that is the actual bandwidth-extension use case.
    """
    down = signal.clone().resample(cutoff_sr).resample(int(signal.sample_rate))
    down.stft_params = TARGET_STFT_PARAMS
    down.stft()
    return down


@torch.no_grad()
@argbind.bind(without_prefix=True)
def _rate_plan(cutoff_rates: list = [8000, 16000, 24000, 32000]):
    return [int(r) for r in cutoff_rates]


def draw_cutoff(step: int, default: int):
    rates = _rate_plan()
    if not rates:
        return default
    g = torch.Generator().manual_seed(0x5A17 + int(step))
    return rates[int(torch.randint(len(rates), (1,), generator=g).item())]


def encode(gen, signal, codec, n_codebooks):
    """Codec-specific encoding. SpectroStream consumes the complex STFT, DAC the waveform."""
    if codec == "dac":
        return gen(signal.audio_data, signal.sample_rate)["codes"][:, :n_codebooks, :]
    return gen(_spectrogram(signal))["codes"][:, :n_codebooks, :]


def _encode_pair(state, batch, accel, cutoff_sr, n_codebooks, dataset):
    """Encode the same audio full-band and band-limited.

    Returns (codes_down, codes_up, signal, rate_idx). rate_idx is the position of
    the cutoff in cutoff_rates; it is only consumed when rate_emb is on.
    """
    codec, _ = arch()
    signal = dataset.transform(batch["signal"].clone(), **batch["transform_args"])
    down = band_limit(signal, cutoff_sr)
    gen = accel.unwrap(state.generator)
    codes_up = encode(gen, signal, codec, n_codebooks)
    codes_down = encode(gen, down, codec, n_codebooks)
    rates = _rate_plan()
    rate_idx = rates.index(cutoff_sr) if cutoff_sr in rates else 0
    return codes_down, codes_up, signal, rate_idx


def _step_losses(state, codes_down, codes_up, n_pred_tokens, accel, ss_prob: float = 0.0,
                 rate_idx: int = 0):
    """Loss over every prediction step (one per target codebook).

    ss_prob is the scheduled-sampling rate: the probability that a token fed back
    into the prefix is the model's OWN prediction rather than the ground truth.
    ss_prob=0 is pure teacher forcing, which is what the paper uses. The argmax
    fed back is taken from the logits already computed for this step's loss, so
    there are no extra forward passes.
    """
    n_out = accel.unwrap(state.transformer).ntoken_output
    B, _, T = codes_up.size()
    losses, top1, top10 = [], [], []
    prefix = []          # what is actually fed back, ground truth or prediction
    for step in range(n_pred_tokens):
        # input: all degraded codebooks, plus the target codebooks already
        # "predicted" in earlier steps
        inp = codes_down if step == 0 else torch.cat([codes_down] + prefix, dim=1)
        tgt = codes_up[:, step, :]
        logits = call_transformer(
            state.transformer, inp, torch.tensor([step], device=codes_up.device),
            rate_idx, accel)
        logits = logits.reshape(B * T, n_out)
        tgt = tgt.reshape(B * T)
        losses.append(state.criterion(logits, tgt))
        with torch.no_grad():
            top1.append(multiclass_accuracy(logits, tgt))
            top10.append(multiclass_accuracy(logits, tgt, k=10))
            truth = codes_up[:, step: step + 1, :]
            if ss_prob > 0:
                pred = logits.argmax(dim=-1).reshape(B, 1, T)
                take_pred = torch.rand_like(truth, dtype=torch.float) < ss_prob
                prefix.append(torch.where(take_pred, pred, truth))
            else:
                prefix.append(truth)
    return (
        torch.stack(losses).sum(),
        torch.stack(top1).mean(),
        torch.stack(top10).mean(),
        torch.stack(losses).detach(),
    )


def scheduled_sampling_rate(step, ss_prob: float, ss_start: int, ss_ramp: int):
    """Linear ramp: 0 until ss_start, then up to ss_prob over ss_ramp iterations."""
    if ss_prob <= 0 or ss_ramp <= 0:
        return 0.0
    return float(ss_prob) * min(1.0, max(0.0, (step - ss_start) / ss_ramp))


@timer()
def train_loop(state, batch, accel, cutoff_sr, n_codebooks, n_pred_tokens,
               ss_prob=0.0, ss_start=0, ss_ramp=1):
    state.generator.eval()          # codec is frozen throughout
    state.transformer.train()
    batch = util.prepare_batch(batch, accel.device)
    codes_down, codes_up, _, rate_idx = _encode_pair(
        state, batch, accel, draw_cutoff(state.tracker.step, cutoff_sr),
        n_codebooks, state.train_data
    )
    p = scheduled_sampling_rate(state.tracker.step, ss_prob, ss_start, ss_ramp)
    with accel.autocast():
        loss, a1, a10, per_step = _step_losses(
            state, codes_down, codes_up, n_pred_tokens, accel, ss_prob=p, rate_idx=rate_idx
        )

    state.optimizer.zero_grad()
    accel.backward(loss)
    accel.scaler.unscale_(state.optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(state.transformer.parameters(), 1e3)
    accel.step(state.optimizer)
    state.scheduler.step()
    accel.update()

    out = {
        "loss": loss / max(n_pred_tokens, 1),
        "acc/top1": a1,
        "acc/top10": a10,
        "other/ss_prob": p,
        "other/grad_norm": grad_norm,
        "other/lr": state.optimizer.param_groups[0]["lr"],
        "loss/first_cb": per_step[0],
        "loss/last_cb": per_step[-1],
    }
    return {k: v for k, v in sorted(out.items())}


@timer()
@torch.no_grad()
def val_loop(batch, state, accel, cutoff_sr, n_codebooks, n_pred_tokens):
    state.generator.eval()
    state.transformer.eval()
    batch = util.prepare_batch(batch, accel.device)
    codes_down, codes_up, _, rate_idx = _encode_pair(
        state, batch, accel, cutoff_sr, n_codebooks, state.val_data
    )
    loss, a1, a10, _ = _step_losses(state, codes_down, codes_up, n_pred_tokens, accel,
                                    rate_idx=rate_idx)
    # Free-running accuracy: the model conditioned on its OWN predictions, i.e.
    # what inference actually does.
    free = predict_codes(state, codes_down, n_pred_tokens, accel, rate_idx=rate_idx)
    a1_free = (free == codes_up[:, :n_pred_tokens, :]).float().mean()
    return {
        "loss": loss / max(n_pred_tokens, 1),
        "acc/top1": a1,
        "acc/top10": a10,
        "acc/top1_free": a1_free,
        "acc/retention": a1_free / a1.clamp_min(1e-9) if torch.is_tensor(a1) else a1_free / max(a1, 1e-9),
    }


@torch.no_grad()
def predict_codes(state, codes_down, n_pred_tokens, accel, temperature: float = 0.0,
                  rate_idx: int = 0):
    """Autoregressive over codebooks (no teacher forcing) -- what inference does."""
    preds = []
    for step in range(n_pred_tokens):
        inp = codes_down if not preds else torch.cat([codes_down] + preds, dim=1)
        logits = call_transformer(
            state.transformer, inp, torch.tensor([step], device=codes_down.device),
            rate_idx, accel)
        if temperature <= 0:
            nxt = logits.argmax(dim=-1)
        else:
            B, T, V = logits.shape
            p = torch.softmax(logits.reshape(-1, V) / temperature, dim=-1)
            nxt = torch.multinomial(p, 1).reshape(B, T)
        preds.append(nxt.unsqueeze(1))
    return torch.cat(preds, dim=1)


@torch.no_grad()
def save_samples(state, val_idx, writer, accel, cutoff_sr, n_codebooks, n_pred_tokens):
    """TensorBoard audio previews. SpectroStream only: it reconstructs from a
    complex spectrogram, and DAC has no such path. Sample logging is a
    convenience; every reported number comes from scripts/evaluate.py on written
    files."""
    codec, _ = arch()
    if codec != "spectrostream":
        return
    state.generator.eval()
    state.transformer.eval()
    samples = [state.val_data[i] for i in val_idx]
    batch = util.prepare_batch(state.val_data.collate(samples), accel.device)
    codes_down, codes_up, signal, rate_idx = _encode_pair(
        state, batch, accel, cutoff_sr, n_codebooks, state.val_data
    )
    pred = predict_codes(state, codes_down, n_pred_tokens, accel, rate_idx=rate_idx)
    gen = accel.unwrap(state.generator)
    length = _spectrogram(signal).shape[-1]

    def to_audio(codes):
        spec = gen.decode_from_codes(codes, length=length)
        rec = signal.clone()
        nyq = torch.zeros(spec.shape[0], spec.shape[1], 1, spec.shape[3],
                          device=spec.device, dtype=spec.dtype)
        rec.stft_data = torch.view_as_complex(
            rearrange(torch.cat([spec, nyq], dim=2), "b c f t -> b 1 f t c").contiguous().float()
        )
        rec.istft()
        rec.ensure_max_of_audio(1.0)
        return rec

    audio = {
        "predicted": to_audio(pred),          # bandwidth-extended
        "input_lowband": band_limit(signal, cutoff_sr),
        "codec_ceiling": to_audio(codes_up),  # codec's own reconstruction: the upper bound
    }
    if state.tracker.step == 0:
        audio["reference"] = signal
    for name, sig in audio.items():
        for nb in range(sig.batch_size):
            sig[nb].cpu().write_audio_to_tb(f"{name}/sample_{nb}.wav", writer, state.tracker.step)


def load_codec(codec, codec_ckpt, args, accel, tracker):
    """Load and freeze the tokenizer.

    SpectroStream: a generator.pth written by scripts/train_codec.py.
    DAC: a run folder holding dac/weights.pth, which carries its own
    hyperparameters, so nothing has to be restated in the config.
    """
    if codec == "dac":
        from dac_codec.model import DAC    # the vendored DAC code (see NOTICE.md)
        # package=False rebuilds the model from dac/weights.pth + dac/metadata.pth
        # using the code in this repository, so the distributed checkpoint does
        # not have to carry a torch.package copy of it. Verified to give
        # bit-identical parameters and codes to the packaged checkpoint.
        generator, _ = DAC.load_from_folder(folder=codec_ckpt, map_location="cpu",
                                            package=False)
        tracker.print(f"Loaded frozen DAC from {codec_ckpt}: "
                      f"{generator.n_codebooks} codebooks, hop {generator.hop_length}")
    else:
        with argbind.scope(args):
            generator = SpS(test=None)
        ck = torch.load(codec_ckpt, map_location="cpu")
        generator.load_state_dict(ck["model.pth"])
        tracker.print(f"Loaded frozen codec from {codec_ckpt} (step {ck['tracker.pth']['step']})")
    generator = generator.to(accel.device).eval()
    for p in generator.parameters():          # frozen: only the transformer trains
        p.requires_grad_(False)
    return generator


@argbind.bind(without_prefix=True)
def load_bwe(args, accel, tracker, save_path, codec_ckpt: str = None,
             resume: bool = False, tag: str = "latest",
             lr_override: float = None):
    codec, _ = arch()
    assert codec_ckpt is not None, "codec_ckpt must point at a trained codec"
    generator = load_codec(codec, codec_ckpt, args, accel, tracker)

    transformer, extra = None, {}
    if resume:
        path = Path(save_path) / tag / "transformer.pth"
        if path.exists():
            transformer = new_transformer()
            extra = torch.load(path, map_location="cpu")
            transformer.load_state_dict(extra["model.pth"])
            tracker.print(f"Resuming transformer from {path}")
    transformer = new_transformer() if transformer is None else transformer
    transformer = accel.prepare_model(transformer)
    tracker.print(transformer)

    with argbind.scope(args, "transformer"):
        optimizer = AdamW(transformer.parameters())
    if "optimizer.pth" in extra:
        optimizer.load_state_dict(extra["optimizer.pth"])

    # lr_override exists for fine-tuning a finished run. Without it, resuming a
    # finished checkpoint trains at eta_min: load_state_dict restores param_groups
    # (lr included) and the checkpoint's CosineAnnealingLR has already annealed.
    # Setting param_groups["lr"] alone is not enough, because the scheduler
    # recomputes lr from base_lrs on its next step(); so the scheduler is built
    # AFTER the override and its state is deliberately not restored.
    if lr_override is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = lr_override
            pg.pop("initial_lr", None)     # let the fresh scheduler re-seed it
        tracker.print(f"lr_override: optimizer set to {lr_override:g}, "
                      f"scheduler rebuilt (checkpoint scheduler state discarded)")

    with argbind.scope(args, "transformer"):
        scheduler = ScheduleLR(optimizer)
    if "scheduler.pth" in extra and lr_override is None:
        scheduler.load_state_dict(extra["scheduler.pth"])
    if "tracker.pth" in extra:
        tracker.load_state_dict(extra["tracker.pth"])

    with argbind.scope(args, "train"):
        train_data = build_dataset()
    with argbind.scope(args, "val"):
        val_data = build_dataset()

    return State(
        generator=generator, transformer=transformer, optimizer=optimizer,
        scheduler=scheduler, criterion=torch.nn.CrossEntropyLoss(),
        train_data=train_data, val_data=val_data, tracker=tracker,
    )


def checkpoint(state, save_iters, save_path, accel):
    tags = ["latest"]
    if "val" in state.tracker.history and state.tracker.is_best("val", "loss"):
        state.tracker.print("Best transformer so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")
    for tag in tags:
        folder = Path(save_path) / tag
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model.pth": {k.replace("_orig_mod.", ""): v
                              for k, v in accel.unwrap(state.transformer).state_dict().items()},
                "optimizer.pth": state.optimizer.state_dict(),
                "scheduler.pth": state.scheduler.state_dict(),
                "tracker.pth": state.tracker.state_dict(),
            },
            folder / "transformer.pth",
        )


def validate(state, val_dataloader, accel, cutoff_sr, n_codebooks, n_pred_tokens):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel, cutoff_sr, n_codebooks, n_pred_tokens)
    return output


@argbind.bind(without_prefix=True)
def train_bwe(
    args,
    accel,
    save_path: str = "runs/bwe",
    num_iters: int = 100000,
    save_iters: list = [5000, 10000, 25000, 50000, 100000],
    valid_freq: int = 500,
    sample_freq: int = 2000,
    batch_size: int = 16,
    val_batch_size: int = 16,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3],
    cutoff_sr: int = 16000,
    n_codebooks: int = 48,
    n_pred_tokens: int = 48,
    # Scheduled sampling. 0.0 is pure teacher forcing, which is what the paper uses.
    ss_prob: float = 0.0,
    ss_start: int = 0,
    ss_ramp: int = 1,
):
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    tracker = Tracker(writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank)

    codec, rate_emb = arch()
    tracker.print(f"codec={codec}  cutoff_rates={_rate_plan()}  rate_emb={rate_emb}")
    state = load_bwe(args, accel, tracker, save_path)
    train_dl = accel.prepare_dataloader(
        state.train_data, start_idx=state.tracker.step * batch_size, num_workers=num_workers,
        batch_size=batch_size, collate_fn=state.train_data.collate,
        persistent_workers=num_workers > 0,
    )
    train_dl = get_infinite_loader(train_dl)
    val_dl = accel.prepare_dataloader(
        state.val_data, start_idx=0, num_workers=num_workers, batch_size=val_batch_size,
        collate_fn=state.val_data.collate, persistent_workers=num_workers > 0,
    )

    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dl))(val_loop)
    validate = tracker.log("val", "mean")(validate)
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dl, start=tracker.step):
            train_loop(state, batch, accel, cutoff_sr, n_codebooks, n_pred_tokens,
                       ss_prob=ss_prob, ss_start=ss_start, ss_ramp=ss_ramp)
            last = tracker.step == num_iters - 1
            if tracker.step % sample_freq == 0 or last:
                save_samples(state, val_idx, writer, accel, cutoff_sr, n_codebooks, n_pred_tokens)
            if tracker.step % valid_freq == 0 or last:
                validate(state, val_dl, accel, cutoff_sr, n_codebooks, n_pred_tokens)
                checkpoint(state, save_iters, save_path, accel)
                tracker.done("val", f"Iteration {tracker.step}")
            if last:
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train_bwe(args, accel)
