__version__ = "1.0.0"

import audiotools

audiotools.ml.BaseModel.INTERN += ["spectrostream.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]

from . import nn
from . import model
from .model import SpS
from .model import SpSFile
