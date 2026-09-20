# Attribution

This repository's structure and much of its plumbing are **inherited from
Descript's Audio Codec (DAC)**, MIT licensed:

- <https://github.com/descriptinc/descript-audio-codec>
- <https://github.com/descriptinc/audiotools> (`descript-audiotools`)

Specifically:

- `spectrostream/` follows DAC's package layout (`model/`, `nn/`) and reuses its building
  blocks: `spectrostream/nn/layers.py`, `spectrostream/nn/quantize_EMA.py`, `spectrostream/nn/loss.py` and
  `spectrostream/model/base.py` are DAC files, adapted where the spectral-domain codec
  needs different shapes. Docstrings in those files name the upstream source.
- `bwe/transformer.py` is a port of the token-prediction transformer we
  used in earlier work, itself derived from DAC's `transup` model.
- The training scripts follow DAC's `argbind` + `audiotools` conventions:
  configuration by YAML with `$include`, `Accelerator`, `Tracker`, dataloaders
  and transforms all come from `audiotools`.
- `scripts/evaluate.py` follows DAC's evaluation script; ViSQOL, SI-SDR, the
  multi-scale STFT and mel distances are `audiotools`/DAC implementations.

**SpectroStream** (`spectrostream/model/spectrostream.py`) is our own reimplementation of

> Y. Li, K. Han, B. McWilliams, Z. Borsos, M. Tagliasacchi,
> "SpectroStream: A versatile neural codec for general audio", arXiv:2508.05207, 2025.

No public implementation exists; the architecture follows the paper, and any
error in it is ours.

`dac_codec/` is **DAC's own code**, MIT licensed, vendored unchanged from the copy
used to train the DAC codec of the paper (`scripts/codec/train_dac.py` is its
training script). It is included so that the exact code behind the released DAC
checkpoint, and behind the mel metric reported in the paper, is in this
repository rather than assumed from a pip package.

**The DAC codec used in the paper was retrained by us** at 48 kHz with 12
codebooks (`conf/codec/dac_48khz.yml`); it is not a Descript-released
checkpoint.

The baselines compared in the paper (A2SB, UniverSR) are not redistributed; see
their own repositories.
