import gzip
import json
import logging
import random

from mda.data.data_collection import DataCollection, DataSample
from mda.util import get_full_path, read_txt_as_str_list
from tqdm import tqdm

from . import INGESTOR_REGISTRY, Ingestor

logger = logging.getLogger(__name__)

RAW_DIR = "data_raw/amazon"
CATEGORIES = [
    "Clothing_Shoes_and_Jewelry",
    "Electronics",
    "Home_and_Kitchen",
    "Kindle_Store",
    "Movies_and_TV",
]
RATING_NAMES = ["low", "medium", "high"]

SUBSAMPLE_PROP = 0.002  # proportion of all sample to keep
REQUIRED_KEYS = ["overall", "reviewTime", "reviewText"]


def rating_to_ridx(rating: float) -> int:
    # 1:low
    # 2-4:medium
    # 5:high

    assert int(rating) == rating
    assert rating >= 1.0 and rating <= 5.0
    if rating == 1.0:
        return 0
    elif rating in [2.0, 3.0, 4.0]:
        return 1
    elif rating == 5.0:
        return 2
    else:
        raise NotImplementedError()


@INGESTOR_REGISTRY.register("amazon")
class AmazonReviewIngestor(Ingestor):
    def run(self) -> DataCollection:
        collection = DataCollection()

        for category_str in CATEGORIES:
            jsonl_gz_path = get_full_path(f"{RAW_DIR}/{category_str}_5.json.gz")
            logger.info(f"loading {jsonl_gz_path}")
            with gzip.open(jsonl_gz_path, "r") as g:
                lines = [l for l in g]
            logger.info(f" loaded {jsonl_gz_path}")

            n_lines = len(lines)
            n_samples_to_keep = int(n_lines * SUBSAMPLE_PROP)
            logger.info(
                f"keeping {n_samples_to_keep} from a total of {n_lines} samples"
            )

            samples = {}
            while len(samples) < n_samples_to_keep:
                idx = random.randint(0, n_lines)
                l = lines[idx]
                s = json.loads(l)
                if not all(k in s for k in REQUIRED_KEYS):
                    continue
                sample_id = f"{s['asin']}.{s['reviewerID']}"
                class_idx = rating_to_ridx(s["overall"])
                sample = DataSample(
                    id=sample_id,
                    text=s["reviewText"],
                    domain_str=category_str,
                    class_str=RATING_NAMES[class_idx],
                    class_idx=class_idx,
                )
                samples[sample_id] = sample

            for sample in samples.values():
                collection.add_sample(sample)

        collection.class_strs = RATING_NAMES
        collection.domain_strs = CATEGORIES
        collection.populate_random_split()
        collection.populate_class_distribution()

        return collection


# example json line
# {
#   "reviewerID": "A2SUAM1J3GNN3B",
#   "asin": "0000013714",
#   "reviewerName": "J. McDonald",
#   "vote": 5,
#   "style": {
#     "Format:": "Hardcover"
#   },
#   "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
#   "overall": 5.0,
#   "summary": "Heavenly Highway Hymns",
#   "unixReviewTime": 1252800000,
#   "reviewTime": "09 13, 2009"
# }
