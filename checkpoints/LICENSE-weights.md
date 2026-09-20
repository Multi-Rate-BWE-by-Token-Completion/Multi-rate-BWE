# Licence of the released model weights

The checkpoints attached to the GitHub release (the SpectroStream codec, the DAC
codec we retrained, and the two multi-rate bandwidth-extension predictors) are
released under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**:
<https://creativecommons.org/licenses/by/4.0/>.

You may use, share and adapt them, including commercially, provided you give
appropriate credit — cite the paper (see CITATION.cff) and link back to this
repository.

The **code** in this repository is MIT licensed instead; see LICENSE.

## What the models were trained on

- SpectroStream and DAC codecs: Jamendo (MTG-Jamendo) and the MUSDB18 training split.
- Bandwidth-extension predictors: Jamendo (filtered to files whose effective
  bandwidth reaches 19 kHz), MedleyDB, the MUSDB18 training split and ENST-Drums.

Those corpora carry their own terms, and MUSDB18 in particular is distributed for
research use. The weights are model parameters, not redistributions of that audio,
but anyone building on them should check the corpora's terms for their own use case.
