from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from ..registry import FromConfigBase, Registry, import_all


class Model(nn.Module, FromConfigBase):
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def to_logdir(self, logdir: str):
        raise NotImplementedError

    def from_logdir(self, log_dir: str):
        raise NotImplementedError


MODEL_REGISTRY = Registry(Model)
import_all(Path(__file__).parent, "mda.model")
