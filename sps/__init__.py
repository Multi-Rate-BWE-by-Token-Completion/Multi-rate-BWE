__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

import audiotools

audiotools.ml.BaseModel.INTERN += ["sps.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn
from . import model
from . import utils
from .model import SpS
from .model import SpSFile