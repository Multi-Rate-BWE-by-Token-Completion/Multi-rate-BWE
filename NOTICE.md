# Attribution

This repository's structure and much of its plumbing are **inherited from
Descript's Audio Codec (DAC)**, MIT licensed:

- <https://github.com/descriptinc/descript-audio-codec>
- <https://github.com/descriptinc/audiotools> (`descript-audiotools`)

Specifically:

- `sps/` follows DAC's package layout (`model/`, `nn/`) and reuses its building
  blocks: `sps/nn/layers.py`, `sps/nn/quantize.py`, `sps/nn/loss.py` and
  `sps/model/base.py` are DAC files, adapted where the spectral-domain codec
  needs different shapes. Docstrings in those files name the upstream source.
- `sps/model/transformer.py` is a port of the token-prediction transformer we
  used in earlier work, itself derived from DAC's `transup` model.
- The training scripts follow DAC's `argbind` + `audiotools` conventions:
  configuration by YAML with `$include`, `Accelerator`, `Tracker`, dataloaders
  and transforms all come from `audiotools`.
- `scripts/evaluate.py` follows DAC's evaluation script; ViSQOL, SI-SDR, the
  multi-scale STFT and mel distances are `audiotools`/DAC implementations.

**SpectroStream** (`sps/model/sps.py`) is our own reimplementation of

> Y. Li, K. Han, B. McWilliams, Z. Borsos, M. Tagliasacchi,
> "SpectroStream: A versatile neural codec for general audio", arXiv:2508.05207, 2025.

No public implementation exists; the architecture follows the paper, and any
error in it is ours.

**The DAC codec used in the paper** is upstream DAC (`descript-audio-codec`
1.0.0) with the architecture in `conf/codec/dac_48khz.yml` (48 kHz, 12
codebooks). It is loaded here through the upstream package.

The baselines compared in the paper (A2SB, UniverSR) are not redistributed; see
their own repositories.
