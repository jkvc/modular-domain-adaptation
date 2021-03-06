import logging
from typing import Tuple

from mda.data.data_collection import DataCollection, DataSample
from mda.util import load_json
from repo_root import get_full_path

from . import INGESTOR_REGISTRY, Ingestor

logger = logging.getLogger(__name__)

RAW_DIR = "data_raw/mfc"
ISSUES = [  # domains
    "climate",
    "deathpenalty",
    "guncontrol",
    "immigration",
    "samesex",
    "tobacco",
]
PRIMARY_FRAME_NAMES = [  # class names
    "Economic",
    "Capacity and Resources",
    "Morality",
    "Fairness and Equality",
    "Legality, Constitutionality, Jurisdiction",
    "Policy Prescription and Evaluation",
    "Crime and Punishment",
    "Security and Defense",
    "Health and Safety",
    "Quality of Life",
    "Cultural Identity",
    "Public Sentiment",
    "Political",
    "External Regulation and Reputation",
    "Other",
]


def _primary_frame_code_to_class_idx(frame_float: float) -> int:
    # see codes.json, non null frames are [1.?, 15.?], map them to [0, 14]
    assert frame_float != 0
    assert frame_float < 16
    return int(frame_float) - 1


@INGESTOR_REGISTRY.register("mfc")
class MediaFrameCorpusIngestor(Ingestor):
    def run(self) -> Tuple[DataCollection, DataCollection]:
        collection = DataCollection()
        all_train_ids = set()
        all_test_ids = set()

        for issue in ISSUES:
            logger.info(issue)
            raw_dict = load_json(get_full_path(f"{RAW_DIR}/{issue}_labeled.json"))

            test_ids = set(
                load_json(get_full_path(f"{RAW_DIR}/{issue}_test_sets.json"))[
                    "primary_frame"
                ]
            )
            train_ids = set(
                id
                for id, item in raw_dict.items()
                if (
                    id not in test_ids
                    and item["primary_frame"] != 0
                    and item["primary_frame"] != None
                )
            )
            all_train_ids = all_train_ids.union(train_ids)
            all_test_ids = all_test_ids.union(test_ids)

            for id in train_ids.union(test_ids):
                item = raw_dict[id]
                class_idx = _primary_frame_code_to_class_idx(item["primary_frame"])
                cleaned_text = "\n".join(item["text"].split("\n\n")[2:])
                sample = DataSample(
                    id=id,
                    text=cleaned_text,
                    domain_str=issue,
                    class_str=PRIMARY_FRAME_NAMES[class_idx],
                    class_idx=class_idx,
                )
                collection.add_sample(sample)

        train_collection = DataCollection()
        for id in all_train_ids:
            train_collection.add_sample(collection.samples[id])
        train_collection.class_strs = PRIMARY_FRAME_NAMES
        train_collection.domain_strs = ISSUES
        train_collection.populate_class_distribution()

        test_collection = DataCollection()
        for id in all_test_ids:
            test_collection.add_sample(collection.samples[id])
        test_collection.class_strs = PRIMARY_FRAME_NAMES
        test_collection.domain_strs = ISSUES
        test_collection.populate_class_distribution()

        return train_collection, test_collection
