from pathlib import Path

from mda.registry import FromConfigBase, Registry, import_all
from omegaconf import OmegaConf


class Workflow(FromConfigBase):
    def run(config: OmegaConf):
        raise NotImplementedError()


WORKFLOW_REGISTRY = Registry(Workflow)
import_all(Path(__file__).parent, "experiments.workflow")
