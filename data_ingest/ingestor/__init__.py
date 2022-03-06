from pathlib import Path

from mda.registry import FromConfigBase, Registry, import_all


class Ingestor(FromConfigBase):
    def run():
        raise NotImplementedError()


INGESTOR_REGISTRY = Registry(Ingestor)
import_all(Path(__file__).parent, "data_ingest.ingestor")
