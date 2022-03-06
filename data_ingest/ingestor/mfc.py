import logging

from mda.data.data_collection import DataCollection
from mda.util import get_full_path, load_json

from . import INGESTOR_REGISTRY, Ingestor

logger = logging.getLogger(__name__)

RAW_DIR = "data_raw/mfc"
ISSUES = [
    "climate",
    "deathpenalty",
    "guncontrol",
    "immigration",
    "police",
    "samesex",
    "tobacco",
]


@INGESTOR_REGISTRY.register("mfc")
class MediaFrameCorpusIngestor(Ingestor):
    def run(self):
        collection = DataCollection()

        for issue in ISSUES:
            logger.info(issue)
            raw_dict = load_json(get_full_path(f"{RAW_DIR}/{issue}_labeled.json"))
