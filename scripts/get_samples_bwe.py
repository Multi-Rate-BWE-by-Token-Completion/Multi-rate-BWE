"""Bandwidth extension synthesis: band-limited codes -> extended audio.

Protocol (Sec. 2.4 of the paper), applied identically to every system compared:

    output = ground-truth low band below the input's cutoff
           + the decoded prediction above it,
      the two joined by a linear crossfade of --xfade bins (20 bins = 1 kHz)
      in the SpectroStream analysis STFT (960/480).

Decoding is argmax (--top_p 0) and reconstructs from the first predicted level
(--n_decode 1) by truncating the RVQ prefix; neither changes the transmitted
rate, which is n_codebooks levels.

Both codecs are supported and only the composition differs:
  spectrostream  decodes to a complex spectrogram, spliced there directly;
  dac            decodes to a waveform, which is then analysed with the same
                 STFT so that the splice is identical.

Conditions written, each as a folder of wavs:
    predicted       the result (n_pred_tokens levels decoded)
    predicted_k{d}  the same prediction truncated to d levels (--n_decode)
    ceiling         the codec's own reconstruction from TRUE full-band codes,
                    i.e. perfect prediction at the same bit rate
    lowband         the band-limited input, no high band: the anchor

The reconstruction is written under the sr48000 name and the low band under the
band-limited name, which is what scripts/evaluate.py expects.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal, STFTParams, ml
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from einops import rearrange

sys.path.append(os.getcwd())
_here = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location("train_bwe", os.path.join(_here, "train_bwe.py"))
tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tb)

STFT_PARAMS = STFTParams(window_length=960, hop_length=480)


def nucleus(logits, top_p: float):
    """Nucleus sampling; top_p <= 0 falls back to argmax (what the paper uses).

    INCLUSIVE cumsum: the mask is `cumulated < top_p`, so the token that carries
    the distribution past top_p is excluded, and mask[..., 0] = True forces the
    top-1 back in.
    """
    probs = torch.softmax(logits, dim=-1)
    if top_p <= 0:
        return probs.argmax(dim=-1)
    ordered, order_idx = torch.sort(probs, dim=-1, descending=True)
    cumulated = torch.cumsum(ordered, dim=-1)
    mask = cumulated < top_p
    mask[..., 0] = True
    masked = torch.where(mask, ordered, torch.zeros_like(ordered))
    masked = torch.nn.functional.normalize(masked, p=1, dim=-1)
    probs = torch.zeros_like(probs).scatter(-1, order_idx, masked)
    V = probs.size(-1)
    return torch.multinomial(probs.reshape(-1, V), num_samples=1).reshape(probs.shape[:-1])


@torch.no_grad()
def predict(tr, codes_down, n_pred, top_p, rate_idx=None):
    """Autoregressive over RVQ depth, as at training time."""
    preds = []
    for step in range(n_pred):
        inp = codes_down if not preds else torch.cat([codes_down] + preds, dim=1)
        st = torch.tensor([step], device=codes_down.device)
        logits = (tr(inp, st, rate_idx=rate_idx) if getattr(tr, "rate_emb", None) is not None
                  else tr(inp, st))
        preds.append(nucleus(logits, top_p).unsqueeze(1))
    return torch.cat(preds, dim=1)


def blend(out, rec, sb, xf):
    """Replace bins >= sb with rec, crossfading over `xf` bins below the boundary.

    xf=0 is a brick wall. The paper uses 20 bins (1 kHz at 50 Hz per bin), chosen
    on mel distance against 1.5 and 2 kHz.
    """
    if xf <= 0:
        out[:, :, sb:, :] = rec[:, :, sb:, :]
        return out
    lo = max(sb - xf, 0)
    w = torch.linspace(0., 1., sb - lo, device=out.device, dtype=out.dtype).view(1, 1, -1, 1)
    out[:, :, lo:sb, :] = (1 - w) * out[:, :, lo:sb, :] + w * rec[:, :, lo:sb, :]
    out[:, :, sb:, :] = rec[:, :, sb:, :]
    return out


def to_audio(spec, reference):
    """(B, 2, F, T) real/imag -> AudioSignal, re-attaching the dropped Nyquist bin."""
    nyq = torch.zeros(spec.shape[0], spec.shape[1], 1, spec.shape[3],
                      device=spec.device, dtype=spec.dtype)
    sig = reference.clone()
    sig.stft_data = torch.view_as_complex(
        rearrange(torch.cat([spec, nyq], dim=2), "b c f t -> b 1 f t c").contiguous().float())
    sig.istft()
    return sig


def main():
    # add_help=False because argbind parses --help too (it lists everything the
    # config exposes). Handle it here first, so that `--help` prints this
    # script's own options instead of failing on the required --output.
    ap = argparse.ArgumentParser(add_help=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="samples/input_25s_16-48",
                    help="folder of sample_<i>_sr48000.wav references and their "
                         "sample_<i>_sr<cutoff>.wav band-limited partners")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--top_p", type=float, default=0.0, help="0 = argmax (the paper)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xfade", type=int, default=20, help="crossfade width in bins; 0 = brick wall")
    ap.add_argument("--tag", default="25k", help="checkpoint tag under save_path")
    ap.add_argument("--n_decode", default="1",
                    help="comma-separated RVQ prefix depths to also render as predicted_k{d}")
    ap.add_argument("--split_bin", type=int, default=0,
                    help="bin above which content comes from the prediction. 0 = derive from "
                         "the input set's sample rate (bin = cutoff_hz / 100)")
    ap.add_argument("--rate_idx", default="",
                    help="cutoff conditioning for a predictor trained with rate_emb: "
                         "'' = model takes no rate argument, 'auto' = look the input set's "
                         "cutoff up in cutoff_rates, or an explicit integer")
    ap.add_argument("--gain", choices=["shared", "per_condition"], default=None,
                    help="clip-avoidance gain. shared: one scalar per item across all "
                         "conditions, so they stay on a common scale. per_condition: a "
                         "condition is attenuated only if it clips on its own. Default "
                         "follows the paper: shared for spectrostream, per_condition for dac.")
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        ap.print_help()
        print("\nEverything in the config can also be overridden, e.g. "
              "--codec_ckpt PATH --n_decode 1,2,4")
        raise SystemExit(0)
    known, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    args = argbind.parse_args()
    torch.manual_seed(known.seed)
    with argbind.scope(args), ml.Accelerator(amp=False) as accel:
        tracker = Tracker(writer=None, log_file="/dev/null", rank=0)
        codec, rate_emb = tb.arch()
        gain_mode = known.gain or ("per_condition" if codec == "dac" else "shared")
        # resume=True is required: without it the transformer is built from
        # scratch and every prediction is random. Check the log says
        # "Resuming transformer from" -- "Loaded frozen codec" is a different load.
        state = tb.load_bwe(args, accel, tracker, args.get("save_path"),
                            resume=True, tag=known.tag)
        gen = accel.unwrap(state.generator)
        tr = accel.unwrap(state.transformer).eval()
        ncb = args.get("n_codebooks", 48)
        npt = args.get("n_pred_tokens", 48)
        depths = [int(d) for d in known.n_decode.split(",") if d.strip()]
        assert all(0 < d <= npt for d in depths), depths

        files = sorted(util.find_audio(known.input))
        rf_f = [f for f in files if "sr48000" in f.name]
        rates = sorted({int(f.name.split("_sr")[1].split(".")[0]) for f in files} - {48000})
        assert len(rates) == 1, f"expected one band-limited rate, found {rates}"
        cut = rates[0]
        lo_f = [f for f in files if f"sr{cut}" in f.name]
        assert len(lo_f) == len(rf_f)
        SB = known.split_bin if known.split_bin > 0 else cut // 100

        if known.rate_idx == "":
            ridx = None
        elif known.rate_idx == "auto":
            plan = tb._rate_plan()
            assert cut in plan, f"cutoff {cut} not in cutoff_rates {plan}"
            ridx = plan.index(cut)
        else:
            ridx = int(known.rate_idx)
        # Assert on the BUILT model, not on the config: decoding a rate-conditioned
        # model without --rate_idx (or the reverse) is silent misconditioning.
        has_emb = getattr(tr, "rate_emb", None) is not None
        assert has_emb == (ridx is not None), (
            f"model {'has' if has_emb else 'has no'} rate_emb but --rate_idx={known.rate_idx!r}")
        print(f"codec={codec}  {ncb} in -> {npt} predicted  top_p={known.top_p}  "
              f"cutoff {cut} Hz -> split_bin {SB}  xfade={known.xfade}  gain={gain_mode}  "
              f"rate_idx={ridx}  seed={known.seed}", flush=True)

        conds = ["predicted", "ceiling", "lowband"] + [f"predicted_k{d}" for d in depths]
        outs = {c: Path(known.output) / c for c in conds}
        for d in outs.values():
            d.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for i in range(0, len(rf_f), known.batch_size):
                lb, rb = lo_f[i:i + known.batch_size], rf_f[i:i + known.batch_size]
                # The band-limited signal, resampled to 48 kHz. This is the
                # decoder-side low band: it is kept exactly and never passed
                # through the codec.
                low = AudioSignal.batch([AudioSignal(str(p)) for p in lb]).resample(48000)
                low.stft_params = STFT_PARAMS; low.stft(); low = low.to(accel.device)
                ref = AudioSignal.batch([AudioSignal(str(p)) for p in rb])
                ref.stft_params = STFT_PARAMS; ref.stft(); ref = ref.to(accel.device)
                low_spec = tb._spectrogram(low)
                nf = low_spec.shape[-1]
                n = ref.audio_data.shape[-1]

                # Input codes come from the BAND-LIMITED signal, which is what a
                # decoder holds; the full-band codes are used only for `ceiling`.
                codes_down = tb.encode(gen, low, codec, ncb)
                codes_true = tb.encode(gen, ref, codec, npt)
                codes_pred = predict(tr, codes_down, npt, known.top_p, ridx)

                def compose(codes, depth=None):
                    out = low_spec.clone() if codec == "spectrostream" else low.stft_data.clone()
                    if codes is None:                      # anchor: no high band
                        out[:, :, SB:, :] = 0.0
                        return (to_audio(out, low) if codec == "spectrostream"
                                else _istft_like(out, low, n))
                    c = codes if depth is None else codes[:, :depth, :]
                    if codec == "spectrostream":
                        rec = gen.decode_from_codes(c, length=nf)[..., :nf]
                        return to_audio(blend(out, rec, SB, known.xfade), low)
                    z = gen.quantizer.from_codes(c)[0]
                    wav = gen.decode(z)[..., :n]
                    w = AudioSignal(wav, 48000)
                    w.stft_params = STFT_PARAMS; w.stft()
                    return _istft_like(blend(out, w.stft_data, SB, known.xfade), low, n)

                audio = {"predicted": compose(codes_pred),
                         "ceiling": compose(codes_true),
                         "lowband": compose(None)}
                for d in depths:
                    audio[f"predicted_k{d}"] = compose(codes_pred, depth=d)
                audio = {k: v.cpu() for k, v in audio.items()}

                # Clip-avoidance gain: one scalar per item across all conditions
                # ("shared"), or per condition only if it clips ("per_condition").
                peaks = torch.stack([a.audio_data.abs().amax(dim=(1, 2)) for a in audio.values()])
                shared_gain = 1.0 / peaks.amax(dim=0).clamp(min=1.0)
                for name, a in audio.items():
                    g = (shared_gain if gain_mode == "shared"
                         else 1.0 / a.audio_data.abs().amax(dim=(1, 2)).clamp(min=1.0))
                    for b, p in enumerate(rb):
                        x = a[b].clone(); x.audio_data = x.audio_data * g[b]
                        x.write(outs[name] / p.name)
                        # The band-limited row, so evaluate.py finds a partner for
                        # every reference. It must be written at its own rate.
                        lo = low[b].clone().cpu()
                        lo.audio_data = lo.audio_data * g[b]
                        lo.resample(cut).write(outs[name] / lb[b].name)
                if (i // known.batch_size) % 25 == 0:
                    print(f"  {min(i + known.batch_size, len(rf_f))}/{len(rf_f)}", flush=True)
        print(f"wrote {len(rf_f)} pairs to each of {sorted(outs)}", flush=True)


def _istft_like(stft_data, reference, length):
    """DAC path: back to a waveform from the spliced complex STFT."""
    s = reference.clone()
    s.stft_data = stft_data
    s.istft()
    s.audio_data = s.audio_data[..., :length]
    return s


if __name__ == "__main__":
    main()
