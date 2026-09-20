# Multi-rate bandwidth extension by token completion in neural audio codecs

Implementation of

> Benoît Ginies, Olivier Fercoq, Gaël Richard.
> **Multi-rate bandwidth extension by token completion in neural audio codecs.**
> Submitted to IEEE ICASSP 2027.

Audio examples: <https://multi-rate-bwe-by-token-completion.github.io/>

Bandwidth extension is cast as **token completion inside a frozen neural audio
codec**. The decoder already holds the low band, so the predictor only has to
complete the codes above the input's Nyquist frequency. Because the codec always
ingests the same 48 kHz representation, the predictor is the only rate-dependent
component: **one transformer covers 8, 16, 24 and 32 kHz inputs**, with no
bandwidth conditioning — the cutoff is recoverable from the codes themselves.

The same recipe is applied to two codecs that differ exactly in whether a token
carries a notion of frequency: **SpectroStream** (spectral-domain, our
reimplementation) and **DAC** (waveform-domain).

This is the code used for the paper.

## What is here

```
sps/                     SpectroStream: codec, discriminator and the predictor transformer
  model/sps.py             the codec (our reimplementation of arXiv:2508.05207)
  model/transformer.py     predictor, autoregressive over RVQ depth
  model/transformer_rate.py  the same plus a learned cutoff embedding (ablation)
  nn/                      layers, quantizers, losses
dac_codec/               DAC (Descript's Audio Codec), vendored: the exact code that
                         trained our 48 kHz DAC, and the source of the reported mel
scripts/
  train_codec.py         train the SpectroStream codec
  train_codec_dac.py     retrain the DAC codec at 48 kHz (our codec, not a release)
  train_bwe.py           train a bandwidth-extension predictor (all four arms)
  get_samples_bwe.py     synthesis: band-limited codes -> extended audio
  evaluate.py            objective metrics (ViSQOL, mel, STFT, waveform, SI-SDR)
  make_testsets.py       build the evaluation excerpts and band-limited partners
  make_filelists.py      build training file lists, with the bandwidth filter
  download_checkpoints.py  fetch the released weights
  export_checkpoints.py    how those weights were produced
conf/
  codec/spectrostream_48khz.yml   the SpectroStream codec of the paper
  codec/dac_48khz.yml             the DAC codec we retrained
  bwe/spectrostream_multirate.yml the proposed model
  bwe/spectrostream_per_rate.yml  one predictor per rate (Table 3)
  bwe/spectrostream_rate_emb.yml  + explicit cutoff embedding (Table 3)
  bwe/dac_multirate.yml           the same predictor on DAC
checkpoints/MANIFEST.json  what the release carries, with checksums
```

Trained models are distributed through the GitHub release (see below). No data or
results are included.

## Install

```bash
conda create -n mrbwe python=3.11 && conda activate mrbwe
pip install -r requirements.txt
```

`visqol` is only needed for `scripts/evaluate.py`; it ships as a wheel for some
platforms and otherwise has to be built from
<https://github.com/google/visqol>. Everything else installs from PyPI. The DAC
code is vendored in `dac_codec/`, so the `descript-audio-codec` package is not
required.

Configuration uses [argbind](https://github.com/pseeth/argbind): every value in a
YAML file can be overridden on the command line (`--batch_size 8`), and
`$include` composes files.

## Data

The codecs are trained on Jamendo and the MUSDB18 training split. The predictors
are trained on Jamendo (bandwidth-filtered), MedleyDB, the MUSDB18 training split
and ENST-Drums, weighted 0.766 / 0.106 / 0.064 / 0.064.

```bash
python scripts/make_filelists.py --outdir filelists \
    --source jamendo='/data/jamendo/audio/*/*.mp3' \
    --source medleydb='/data/MedleyDB/train/**/*.wav' \
    --source musdb='/data/musdb18/train/Mixtures/*.wav' \
    --source enst_drums='/data/ENST-drums/train/**/*.wav' \
    --min-cliff jamendo=19 --holdout jamendo=300
```

`--min-cliff jamendo=19` is the filter of the paper: about half of Jamendo is
MP3-sourced and brick-wall lowpassed, so only the files whose effective bandwidth
reaches 19 kHz are kept (39% of the corpus).

Evaluation sets: 1000 excerpts of 2.5 s from the 50 MUSDB18 test tracks (20 per
track), and 1000 OrchideaSOL excerpts out of domain.

```bash
python scripts/make_testsets.py excerpts \
    --source /data/musdb18/test/Mixtures --output samples/input_25s_48 \
    --n 1000 --duration 2.5
python scripts/make_testsets.py rates \
    --src samples/input_25s_48 --dst_prefix samples/input_25s 8000 16000 24000 32000
```

This writes `samples/input_25s_<k>-48/`, each holding `sample_<i>_sr48000.wav`
references and their `sample_<i>_sr<rate>.wav` band-limited partners.

## Training

### 1. The codec

```bash
torchrun --nproc_per_node=4 scripts/train_codec.py \
    --args.load conf/codec/spectrostream_48khz.yml \
    --save_path runs/codec_spectrostream/ --resume --tag latest
```

500k steps, batch 64 over 4 GPUs. The codec is then frozen; the predictors read
`runs/codec_spectrostream/<tag>/generator.pth`.

The DAC codec of the paper is **retrained by us**, not a Descript release: 48 kHz,
12 codebooks (11.25 kbit/s), 44k steps, on the same data.

```bash
python scripts/train_codec_dac.py --args.load conf/codec/dac_48khz.yml \
    --save_path runs/codec_dac_48khz/
```

`conf/bwe/dac_multirate.yml:codec_ckpt` then points at that run folder.

### 2. The predictor

All four arms of the paper are the same script; the config selects the codec, the
input rates and whether a cutoff embedding is used.

```bash
# the proposed model: one predictor for 8/16/24/32 kHz, no rate conditioning
python scripts/train_bwe.py --args.load conf/bwe/spectrostream_multirate.yml \
    --save_path runs/bwe_sps_multirate/ --resume --tag latest

# one predictor per input rate (Table 3, "Per-rate"): one run per rate
python scripts/train_bwe.py --args.load conf/bwe/spectrostream_per_rate.yml \
    --cutoff_rates [8000] --save_path runs/bwe_sps_8k/ --resume --tag latest

# + explicit cutoff embedding (Table 3, "+rate emb.")
python scripts/train_bwe.py --args.load conf/bwe/spectrostream_rate_emb.yml \
    --save_path runs/bwe_sps_rate_emb/ --resume --tag latest

# the same predictor on the DAC codec (Table 1, "Ours (DAC)")
python scripts/train_bwe.py --args.load conf/bwe/dac_multirate.yml \
    --save_path runs/bwe_dac_multirate/ --resume --tag latest
```

The codec stays frozen: only the transformer is trained, teacher-forced, with a
cross-entropy loss summed over RVQ steps. The paper reports the **25k-step**
checkpoint (`--tag 25k`), which the configs save.

The two switches that carry the multi-rate claim are `cutoff_rates` (a list
drawn uniformly per batch; a single entry gives a per-rate model) and `rate_emb`
(false in the proposed model).

## Synthesis and evaluation

```bash
# extend a test set; k=1 decodes the first predicted level (see below)
python scripts/get_samples_bwe.py --args.load conf/bwe/spectrostream_multirate.yml \
    --save_path runs/bwe_sps_multirate/ --tag 25k \
    --input samples/input_25s_16-48 --output samples/out_16k \
    --batch_size 8 --seed 0 --top_p 0 --xfade 20 --n_decode 1

# objective metrics, written to samples/out_16k/predicted_k1/metrics.csv
python scripts/evaluate.py --input samples/input_25s_16-48 \
    --output samples/out_16k/predicted_k1 --n_proc 32
```

For a predictor trained with `rate_emb: true`, add `--rate_idx auto`, which looks
the test set's cutoff up in `cutoff_rates`.

The synthesis protocol (paper Sec. 2.4) keeps the ground-truth low band below the
input's cutoff and splices the decoded prediction above it with a 1 kHz crossfade
(20 bins) in the 960/480 STFT. Decoding is argmax (`--top_p 0`) and reconstructs
from the first predicted RVQ level (`--n_decode 1`); deeper levels did not improve
ViSQOL. Neither changes the transmitted rate, which remains `n_codebooks` levels.

Each run writes four folders: `predicted` (all predicted levels), `predicted_k1`
(the reported output), `ceiling` (the codec's own reconstruction from true
full-band codes, i.e. perfect prediction at the same bit rate) and `lowband` (the
band-limited input, the anchor). `metrics.csv` carries one row per file; the
analysis reads the `sr48000` rows.

Metrics are ViSQOL v3 in audio mode, the mel distance, a multi-resolution STFT
distance, waveform L1, and SI-SDR, which is stored as a loss (negate it for dB).

The reported **mel is DAC's**: `dac_codec/nn/loss.py`, built with its defaults —
two resolutions, window lengths {2048, 512}, {150, 80} mel bins, log10 of the
squared magnitude plus a linear term. It is the same metric DAC and our earlier
work report, so the numbers are comparable with them. It is *not* the
SpectroStream training mel in `sps/nn/loss.py`, which is a different objective and
is never used for evaluation.

## Pretrained models

The models used for the paper are attached to the GitHub release.

| Asset | What it is | Size |
|---|---|---|
| `spectrostream_48khz_500k.pth` | SpectroStream codec, 64 RVQ levels, 500k steps (the paper uses the first 48 = 12 kbit/s) | 132 MB |
| `dac_48khz_12cb` | our DAC codec, 48 kHz, 12 codebooks, 44k steps | 307 MB |
| `bwe_spectrostream_multirate_25k.pth` | the multi-rate predictor on SpectroStream, 25k steps (Table 1, "Ours (SpectroStream)") | 574 MB |
| `bwe_dac_multirate_25k.pth` | the multi-rate predictor on DAC, 25k steps (Table 1, "Ours (DAC)") | 385 MB |

```bash
python scripts/download_checkpoints.py      # verifies MD5 against checkpoints/MANIFEST.json
```

This lays them out as the configs expect:

```
checkpoints/spectrostream_48khz_500k.pth
checkpoints/dac_48khz_12cb/dac/{weights,metadata}.pth
checkpoints/bwe_spectrostream_multirate/25k/transformer.pth
checkpoints/bwe_dac_multirate/25k/transformer.pth
```

Then run the synthesis above with
`--codec_ckpt checkpoints/spectrostream_48khz_500k.pth --save_path checkpoints/bwe_spectrostream_multirate --tag 25k`
(and the DAC pair for the DAC arm).

These are **inference checkpoints**: optimiser and scheduler state are stripped
(`scripts/export_checkpoints.py`), so they load for synthesis and evaluation but
cannot resume training. Decoding from them is bit-identical to decoding from the
full training checkpoints.

## Notes

- **Baselines** (A2SB, UniverSR) are not redistributed. In the paper they are run
  from their own repositories and their outputs are put through the same splice.
- The code is inherited from Descript's Audio Codec and `audiotools`; see
  [NOTICE.md](NOTICE.md). SpectroStream has no public implementation and is
  reimplemented here from the paper.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
