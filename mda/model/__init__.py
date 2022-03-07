from pathlib import Path

import torch.nn as nn
from mda.registry import FromConfigBase, Registry, import_all


class Model(nn.Module, FromConfigBase):
    pass


MODEL_REGISTRY = Registry(Model)
import_all(Path(__file__).parent, "mda.model")
