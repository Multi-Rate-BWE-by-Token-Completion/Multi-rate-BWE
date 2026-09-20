"""The bandwidth-extension predictor: a transformer over RVQ depth.

Codec-agnostic: it consumes and produces integer codes, so the same model runs on
the SpectroStream codec (spectrostream/) and on DAC (dac_codec/).
"""
import audiotools

audiotools.ml.BaseModel.INTERN += ["bwe.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]

from .transformer import TransformerModel
from .transformer_rate import RateTransformerModel
