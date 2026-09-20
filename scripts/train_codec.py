"""SpectroStream codec training (48 kHz).

Our reimplementation of Li et al., "SpectroStream: A versatile neural codec for
general audio" (arXiv:2508.05207); no public implementation exists. The codec is
trained adversarially, then frozen for the bandwidth-extension experiments.

    torchrun --nproc_per_node=4 scripts/train_codec.py \\
        --args.load conf/codec/spectrostream_48khz.yml \\
        --save_path runs/codec_spectrostream/ --resume --tag latest

Structure and plumbing are inherited from Descript's Audio Codec; see NOTICE.
"""
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
import inspect

from typing import List
from typing import Union

import librosa
import numpy as np
import argbind
import torch
from einops import rearrange
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset as BaseAudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from audiotools import STFTParams
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())

import sps


warnings.filterwarnings("ignore", category=UserWarning)

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
#AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Adam = argbind.bind(torch.optim.Adam, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


#@argbind.bind("generator", "discriminator")
#def ExponentialLR(optimizer, gamma: float = 1.0):
#    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

@argbind.bind("generator", "discriminator")
def ConstantLR(optimizer, factor: float = 1.0):
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=factor)


# Models
SpS = argbind.bind(sps.model.SpS)
# SpectroStream's own discriminator (paper Fig. 2): multi-scale, 2D-conv, and
# critically it takes (real, imag, modulus) of the STFT as input. That complex
# input is what makes the feature-matching loss (Eq. 5, weighted lambda_feat=100)
# an implicit phase-alignment objective -- the paper has no explicit phase loss,
# so this is the only thing supervising phase. Discriminator_DAC was tried here
# instead and is kept in the codebase, but it is paired with DAC's *waveform*
# decoder where phase coherence is structural; bolted onto a complex-STFT decoder
# it does not play the same role. See runs/SpS-48_DAC_2 for that experiment.
Discriminator = argbind.bind(sps.model.Discriminator)

# Data
AudioDataset = argbind.bind(BaseAudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

TARGET_STFT_PARAMS = STFTParams(window_length=960, hop_length=480)


def compute_target_stft(signal):
    signal.stft_params = TARGET_STFT_PARAMS
    signal.stft()
    return signal


class STFTAudioDataset(BaseAudioDataset):
    # Note: STFT is intentionally *not* computed in __getitem__. AudioSignal.batch()
    # (used by collate below) rebuilds a fresh AudioSignal from each item's raw
    # audio_data only and does not carry over stft_data, so any STFT computed
    # per-item here would just be discarded and recomputed from scratch in
    # collate anyway -- doing it here as well doubled the FFT work per example
    # for nothing. collate() is the only place stft_data actually needs to be
    # computed, on the already-batched signal, using the same TARGET_STFT_PARAMS.
    @staticmethod
    def collate(list_of_dicts, n_splits=None):
        batch = BaseAudioDataset.collate(list_of_dicts, n_splits=n_splits)

        def _apply_stft(output):
            if isinstance(output, dict):
                if "signal" in output:
                    output["signal"] = compute_target_stft(output["signal"])
                else:
                    for value in output.values():
                        _apply_stft(value)

        _apply_stft(batch)
        return batch

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(sps.nn.loss, filter_fn=filter_fn)



def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


# ---------------------------------------------------------------------------
# Phase-coherence diagnostics.
#
# The decoder emits a complex STFT (real, imag), but every reconstruction loss
# we train on -- mel and multi-scale STFT -- is magnitude-only, and the paper
# has no explicit phase term either (its phase supervision is implicit, via
# feature-matching against a discriminator that sees real+imag; see the
# Discriminator binding above). That means nothing in the *logged* losses
# distinguishes "correct magnitude, coherent phase" from "correct magnitude,
# random phase" -- and the latter sounds broken while mel/loss looks healthy.
#
# Measured on this project's history, phase does not begin to emerge until
# ~30k steps and only breaks through around 60-65k, so these are deliberately
# tracked from step 0: the transition is otherwise invisible until it has
# already failed to happen. Reference values on 48kHz music:
#     phase_error ~= 1.5708 (pi/2)  -> indistinguishable from random phase
#     phase_error ~= 0.64, SI-SDR ~= +10 dB -> pretrained DAC
# ---------------------------------------------------------------------------
def phase_error(estimate: AudioSignal, reference: AudioSignal,
                n_fft: int = 960, hop_length: int = 480):
    """Magnitude-weighted mean absolute STFT phase error, in radians.

    Weighting by the *reference* magnitude keeps near-silent bins -- whose phase
    is meaningless and uniformly distributed -- from dominating the average.
    """
    est = estimate.audio_data.reshape(-1, estimate.audio_data.shape[-1])
    ref = reference.audio_data.reshape(-1, reference.audio_data.shape[-1])
    length = min(est.shape[-1], ref.shape[-1])
    window = torch.hann_window(n_fft, device=est.device)
    E = torch.stft(est[..., :length].float(), n_fft, hop_length,
                   window=window, return_complex=True)
    R = torch.stft(ref[..., :length].float(), n_fft, hop_length,
                   window=window, return_complex=True)
    # wrap the difference into [-pi, pi] before taking |.|
    delta = torch.angle(torch.exp(1j * (E.angle() - R.angle())))
    weight = R.abs() / (R.abs().sum() + 1e-9)
    return (weight * delta.abs()).sum()


def si_sdr(estimate: AudioSignal, reference: AudioSignal):
    """Scale-invariant SDR in dB (higher is better).

    Unlike mel/STFT losses this is phase-sensitive: a reconstruction with
    perfect magnitude but random phase scores about -35 dB.
    """
    est = estimate.audio_data.reshape(-1, estimate.audio_data.shape[-1]).float()
    ref = reference.audio_data.reshape(-1, reference.audio_data.shape[-1]).float()
    length = min(est.shape[-1], ref.shape[-1])
    est, ref = est[..., :length], ref[..., :length]
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + 1e-8)
    target = alpha * ref
    noise = est - target
    return (10 * torch.log10(
        (target.pow(2).sum(-1) + 1e-8) / (noise.pow(2).sum(-1) + 1e-8)
    )).mean()


def stft_consistency(stft_data: torch.Tensor, length: int,
                     n_fft: int = 960, hop_length: int = 480):
    """Relative error ||STFT(iSTFT(X)) - X|| / ||X|| of the decoder's raw output.

    Takes the emitted complex field *before* istft -- measuring it after would be
    meaningless, since the STFT of any real waveform is consistent by definition.

    An arbitrary (real, imag) field is not the STFT of any signal: with 50%
    overlap the valid STFTs form a strict subspace. If the decoder emits an
    inconsistent field, overlap-add partially cancels it (destroying energy) and
    the audio smears. ~0 means the output is a genuine STFT; ~0.74 is what a
    random-phase field scores.
    """
    X = stft_data.reshape(-1, stft_data.shape[-2], stft_data.shape[-1])
    window = torch.hann_window(n_fft, device=X.device)
    wav = torch.istft(X, n_fft, hop_length, window=window, length=length)
    Xr = torch.stft(wav, n_fft, hop_length, window=window, return_complex=True)
    T = min(X.shape[-1], Xr.shape[-1])
    return ((Xr[..., :T] - X[..., :T]).abs().pow(2).sum().sqrt()
            / (X[..., :T].abs().pow(2).sum().sqrt() + 1e-9))


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int = 48000,
    folders: dict = None,
    weights: list = None,
):
    # `folders` entries may be directories OR .csv file lists with a "path"
    # column (audiotools' read_sources handles both), which is how the
    # bandwidth-filtered subsets in filelists/ are wired in
    # (see scripts/make_filelists.py).
    #
    # `weights` gives the probability of drawing from each source, in the same
    # order as folders['music_hq']. Without it, sampling is proportional to FILE
    # COUNT, which is not what you usually want: a 3 s clip and a 224 s track get
    # identical draw probability, so a large collection of short files silently
    # dominates. Measured on this corpus, unweighted sampling gave Jamendo 99.8%
    # of draws and MUSDB 0.18%.
    #
    # IMPORTANT: weights are only consulted when AudioDataset.without_replacement
    # is False. When it is True, AudioDataset passes a global_idx and AudioLoader
    # indexes a flat list of every file instead, bypassing the weights entirely --
    # silently, with no error. Set `AudioDataset.without_replacement: false`
    # alongside any weights.
    if weights is not None:
        n_src = len(folders["music_hq"])
        if len(weights) != n_src:
            raise ValueError(
                f"build_dataset: {len(weights)} weights for {n_src} sources"
            )
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("build_dataset: weights must sum to a positive value")
        weights = [w / total for w in weights]

    loader = AudioLoader(sources=folders["music_hq"], weights=weights)
    transform = build_transform()
    dataset = STFTAudioDataset(loader, sample_rate, transform=transform)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: SpS
    optimizer_g: Adam
    scheduler_g: ConstantLR

    discriminator: Discriminator
    optimizer_d: Adam
    scheduler_d: ConstantLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker

    # Running EMA of adv/disc_loss, used by train_loop to decide whether D gets a
    # gradient update this step. Starts at None (first observed value seeds it);
    # not restored from checkpoints on resume, so it re-adapts over the first
    # several dozen steps after any resume -- an acceptable, short transient.
    disc_loss_ema: float = None


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
    compile_model: bool = True,
):

    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    if resume:
        folder = Path(f"{save_path}/{tag}")
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{folder}")

        generator_path = folder / "generator.pth"
        discriminator_path = folder / "discriminator.pth"

        if generator_path.exists():
            generator = SpS(test=None)
            g_extra = torch.load(generator_path, map_location="cpu")
            if "model.pth" in g_extra:
                generator.load_state_dict(g_extra["model.pth"])
        elif (folder / "sps").exists():
            kwargs = {
                "folder": f"{save_path}/{tag}",
                "map_location": "cpu",
                "package": not load_weights,
            }
            generator, g_extra = SpS.load_from_folder(**kwargs)

        if discriminator_path.exists():
            discriminator = Discriminator()
            d_extra = torch.load(discriminator_path, map_location="cpu")
            if "model.pth" in d_extra:
                discriminator.load_state_dict(d_extra["model.pth"])
        elif (folder / "discriminator").exists():
            kwargs = {
                "folder": f"{save_path}/{tag}",
                "map_location": "cpu",
                "package": not load_weights,
            }
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = SpS(test=None) if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(discriminator)

    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    if compile_model:
        # SpS.forward is pure-tensor (no AudioSignal objects touched inside
        # encode/decode), so the whole generator can be compiled as one unit.
        generator = torch.compile(generator)

        # Discriminator.forward is NOT pure-tensor -- it sets signal.stft_params
        # and calls signal.stft() directly on AudioSignal objects, which dynamo
        # can't trace. Only the inner BaseDiscriminator submodules (pure conv/
        # norm/activation tensor ops) are compiled, in place; the outer
        # Discriminator object (and its AudioSignal-handling forward) stays as-is.
        for i in range(len(discriminator.discriminators)):
            discriminator.discriminators[i] = torch.compile(discriminator.discriminators[i])

        # Either way, .train()/.eval()/.parameters()/state_dict() on `generator`
        # (and on `discriminator`'s now-compiled children) proxy correctly to the
        # same underlying parameters -- torch.compile doesn't clone them. The one
        # thing that does change is that state_dict() keys pick up an
        # "_orig_mod." segment wherever compilation wrapped something; that's
        # cleaned up uniformly in checkpoint() below before saving, so resuming
        # from a checkpoint (which always loads into a freshly-constructed,
        # uncompiled model first, see above) is unaffected either way.

    with argbind.scope(args, "generator"):
        optimizer_g = Adam(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ConstantLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = Adam(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ConstantLR(optimizer_d)

    # Capture the LR the *current* config wants, before it can get clobbered by
    # loading a checkpoint saved under a different config (e.g. resuming a run
    # after changing discriminator/Adam.lr) -- optimizer_*.load_state_dict()
    # below restores the LR that was in effect when the checkpoint was saved,
    # which would otherwise silently override this run's new config value.
    configured_lr_g = optimizer_g.param_groups[0]["lr"]
    configured_lr_d = optimizer_d.param_groups[0]["lr"]

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    # Re-apply: keep Adam's loaded momentum/variance state, but force the LR to
    # match the current config rather than whatever was saved in the checkpoint.
    # ConstantLR.get_lr() is a no-op past total_iters (default 5, and we're always
    # resuming well past that), so this sticks and won't get reset by scheduler.step().
    for group in optimizer_g.param_groups:
        group["lr"] = configured_lr_g
    for group in optimizer_d.param_groups:
        group["lr"] = configured_lr_d

    with argbind.scope(args, "train"):
        train_data = build_dataset()
    with argbind.scope(args, "val"):
        val_data = build_dataset()

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)

    signal = state.val_data.transform(batch['signal'].clone(), **batch["transform_args"])

    spectrograms = torch.view_as_real(signal.stft_data)

    spectrograms = rearrange(spectrograms, "b 1 f t c -> b c f t")
    # STFT frequency bins are ordered from 0 Hz to Nyquist, so drop the last bin.
    spectrograms = spectrograms[:, :, :-1, :]

    out = state.generator(
        spectrograms
        )

    recons = signal.clone()
    nyquist_bin = torch.zeros(
        out["spectrogram"].shape[0],
        out["spectrogram"].shape[1],
        1,
        out["spectrogram"].shape[3],
        device=out["spectrogram"].device,
        dtype=out["spectrogram"].dtype,
    )
    spectrogram_with_nyquist = torch.cat([out["spectrogram"], nyquist_bin], dim=2)
    recons.stft_data = torch.view_as_complex(
        rearrange(spectrogram_with_nyquist, "b c f t -> b 1 f t c").contiguous()
    )

    # Keep the decoder's raw complex field before istft() overwrites nothing but
    # is the only point at which consistency is still measurable meaningfully.
    emitted_stft = recons.stft_data.clone()

    recons.istft()

    return {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
        # Phase diagnostics -- none of the losses above can distinguish coherent
        # phase from random phase, so these are the only signal that the model
        # is actually reconstructing the waveform rather than just its envelope.
        "phase/error": phase_error(recons, signal),
        "phase/si_sdr": si_sdr(recons, signal),
        "phase/consistency": stft_consistency(
            emitted_stft, length=signal.audio_data.shape[-1]
        ),
    }


@timer()
def train_loop(state, batch, accel, lambdas, disc_loss_ema_decay=0.99, disc_skip_below_loss=0.0):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)

    with torch.no_grad():

        signal = state.train_data.transform(batch['signal'].clone(), **batch["transform_args"])

        spectrograms = torch.view_as_real(signal.stft_data)

        spectrograms = rearrange(spectrograms, "b 1 f t c -> b c f t")
        # STFT frequency bins are ordered from 0 Hz to Nyquist, so drop the last bin.
        spectrograms = spectrograms[:, :, :-1, :]


    with accel.autocast(dtype=torch.bfloat16):
        out = state.generator(
            spectrograms
        )

        recons = signal.clone()
        nyquist_bin = torch.zeros(
            out["spectrogram"].shape[0],
            out["spectrogram"].shape[1],
            1,
            out["spectrogram"].shape[3],
            device=out["spectrogram"].device,
            dtype=out["spectrogram"].dtype,
        )
        spectrogram_with_nyquist = torch.cat([out["spectrogram"], nyquist_bin], dim=2)
        # cuFFT's half-precision path only supports power-of-two transform sizes;
        # our window_length=960 isn't one, so force this istft() back to float32
        # regardless of autocast. Dimensions/params are unaffected, only precision.
        recons.stft_data = torch.view_as_complex(
            rearrange(spectrogram_with_nyquist, "b c f t -> b 1 f t c").contiguous().float()
        )

        recons.istft()

        commitment_loss = out["vq/commitment_losses"]
        codebook_loss = out["vq/codebook_losses"]
        
    # A static "update D every N steps" throttle can't tell the difference between
    # "D is dominating" (should skip) and "D needs to catch up" (should update) --
    # it was fighting itself: strong enough to stop the ~10k-step collapse, but the
    # same fixed suppression also kept adv/feat_loss completely flat for 70k+
    # steps (verified: D never learns enough to develop features worth matching).
    # Adaptive alternative: always compute disc_loss (needed for logging and the
    # EMA either way), track a running average of it, and only actually give D a
    # gradient update when that average says it isn't already winning. This lets D
    # get *more* updates while it's genuinely behind and fewer once it's ahead,
    # instead of a fixed compromise between the two regimes.
    with accel.autocast(dtype=torch.bfloat16):
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal)

    disc_loss_value = output["adv/disc_loss"].detach().item()
    if state.disc_loss_ema is None:
        state.disc_loss_ema = disc_loss_value
    else:
        state.disc_loss_ema = (
            disc_loss_ema_decay * state.disc_loss_ema
            + (1 - disc_loss_ema_decay) * disc_loss_value
        )
    output["other/disc_loss_ema"] = state.disc_loss_ema

    if state.disc_loss_ema >= disc_skip_below_loss:
        state.optimizer_d.zero_grad()
        accel.backward(output[f"adv/disc_loss"])
        accel.scaler.unscale_(state.optimizer_d)
        output[f"other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
            state.discriminator.parameters(), 10.0
        )
        accel.step(state.optimizer_d)
        state.scheduler_d.step()

    with accel.autocast(dtype=torch.bfloat16):
        output["stft/loss"] = state.stft_loss(recons, signal)
        output["mel/loss"] = state.mel_loss(recons, signal)
        output["waveform/loss"] = state.waveform_loss(recons, signal)
        (
            output["adv/gen_loss"],
            output["adv/feat_loss"],
        ) = state.gan_loss.generator_loss(recons, signal)
        output["vq/commitment_loss"] = commitment_loss
        output["vq/codebook_loss"] = codebook_loss
        output["loss"] = sum([v * output[k] for k, v in lambdas.items() if k in output])

    state.optimizer_g.zero_grad()
    accel.backward(output[f"loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output[f"other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output[f"other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output[f"other/batch_size"] = signal[0].batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def _compile_clean_state_dict(module):
    # torch.compile() wraps modules in an OptimizedModule whose state_dict()
    # keys pick up an "_orig_mod." segment wherever compilation was applied
    # (either at the top level, e.g. the generator, or nested, e.g. each
    # compiled BaseDiscriminator inside discriminator.discriminators). Strip
    # it so checkpoints always use the same clean key names regardless of
    # whether compile_model was on, and can be loaded into a fresh, uncompiled
    # model either way.
    return {k.replace("_orig_mod.", ""): v for k, v in module.state_dict().items()}


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", f"mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append(f"best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        folder = Path(save_path) / tag
        folder.mkdir(parents=True, exist_ok=True)

        generator_extra = {
            "model.pth": _compile_clean_state_dict(accel.unwrap(state.generator)),
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        torch.save(generator_extra, folder / "generator.pth")

        discriminator_extra = {
            "model.pth": _compile_clean_state_dict(accel.unwrap(state.discriminator)),
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        torch.save(discriminator_extra, folder / "discriminator.pth")


@torch.no_grad()
def save_samples(state, val_idx, writer):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]

    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)

    signal = state.val_data.transform(batch['signal'].clone(), **batch["transform_args"])

    spectrograms = torch.view_as_real(signal.stft_data)

    spectrograms = rearrange(spectrograms, "b 1 f t c -> b c f t")
    # STFT frequency bins are ordered from 0 Hz to Nyquist, so drop the last bin.
    spectrograms = spectrograms[:, :, :-1, :]

    out = state.generator(
            spectrograms
        )

    recons = signal.clone()
    nyquist_bin = torch.zeros(
        out["spectrogram"].shape[0],
        out["spectrogram"].shape[1],
        1,
        out["spectrogram"].shape[3],
        device=out["spectrogram"].device,
        dtype=out["spectrogram"].dtype,
    )
    spectrogram_with_nyquist = torch.cat([out["spectrogram"], nyquist_bin], dim=2)
    recons.stft_data = torch.view_as_complex(
        rearrange(spectrogram_with_nyquist, "b c f t -> b 1 f t c").contiguous()
    )

    recons.istft()

    # The decoder's raw output isn't guaranteed to stay within [-1, 1] (no bounded
    # output activation, unlike the training target which is peak-safe via
    # RescaleAudio) -- torch's add_audio() hard-clips any sample outside that
    # range before writing the WAV, which introduces real digital-clipping
    # distortion into what we listen to that was never part of training (all
    # losses are computed on the raw float tensors, never through this clamp).
    # Peak-rescale here instead: uniformly scales down only if needed, preserving
    # waveform shape, so listening reflects the actual learned reconstruction.
    recons.ensure_max_of_audio(1.0)

    audio_dict = {
        "recons": recons,
    }
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )



def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 72,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    disc_loss_ema_decay: float = 0.99,
    disc_skip_below_loss: float = 0.0,
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    }
):
    #util.seed(seed)

    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    state = load(args, accel, tracker, save_path)

    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):

            train_loop(state, batch, accel, lambdas, disc_loss_ema_decay, disc_skip_below_loss)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, val_idx, writer)

            if tracker.step % valid_freq == 0 or last_iter:
                validate(state, val_dataloader, accel)
                checkpoint(state, save_iters, save_path)
                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel) 

"""
{
    'idx': 12, 
    'transform_args': {
        'Compose': {
            '0.preprocess': {
                '0.Identity': {'mask': tensor(True)}, 
                'mask': tensor(True)
                }, 
            '1.augment': {
                '0.Identity': {'mask': tensor(True)}, 
                'mask': tensor(False)
                }, 
            '2.postprocess': {
                '0.VolumeNorm': {'db': tensor(-16), 'mask': tensor(True)}, 
                '1.RescaleAudio': {'mask': tensor(True)}, 
                '2.ShiftPhase': {'shift': tensor(-2.9316), 'mask': tensor(True)}, 
                'mask': tensor(True)
                }, 
            'mask': tensor(True)
            }
        }, 
    'signal': <audiotools.core.audio_signal.AudioSignal object at 0x7fcd4cda4bd0>, 
    'source_idx': 1, 
    'item_idx': 13, 
    'source': '/path/to/musdb18/train/Mixtures', 
    'path': '/path/to/musdb18/train/Mixtures/<track>.wav'
    }
"""