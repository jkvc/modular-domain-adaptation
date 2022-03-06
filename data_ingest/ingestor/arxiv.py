import json
import logging

from mda.data.data_collection import DataCollection, DataSample
from mda.util import get_full_path, load_json, read_txt_as_str_list
from tqdm import tqdm

from . import INGESTOR_REGISTRY, Ingestor

logger = logging.getLogger(__name__)

RAW_PATH = "data_raw/arxiv/arxiv-metadata-oai-snapshot.json"
ARXIV_CATEGORIES = [  # domains
    "cs.AI",
    "cs.CL",  # computation and language
    "cs.CV",
    "cs.LG",  # machine learning
    "cs.NE",  # neural
    "cs.SI",  # social and information network
]
YEARRANGE2BOUNDS = {  # keys: class names
    "upto2008": (0, 2008),
    "2009-2014": (2009, 2014),
    "2015-2018": (2015, 2018),
    "2019after": (2019, 6969),
}
CLASS_STRS = list(YEARRANGE2BOUNDS.keys())


def _year2yidx(year: int) -> int:
    for i, yearrange_name in enumerate(CLASS_STRS):
        lb, ub = YEARRANGE2BOUNDS[yearrange_name]
        if year >= lb and year <= ub:
            return i
    raise ValueError()


@INGESTOR_REGISTRY.register("arxiv")
class ArxivIngestor(Ingestor):
    def run(self) -> DataCollection:
        collection = DataCollection()

        raw_full_path = get_full_path(RAW_PATH)
        logger.info(f"loading {raw_full_path}")
        lines = read_txt_as_str_list(raw_full_path)
        logger.info(f"loaded {len(lines)} lines of json")

        for line in tqdm(lines):
            raw_dict = json.loads(line)
            categories = raw_dict["categories"].split(" ")
            categories = [c for c in categories if c in ARXIV_CATEGORIES]
            if not categories:
                continue
            category = categories[0]

            if "abstract" not in raw_dict or len(raw_dict["abstract"]) == 0:
                continue

            id = raw_dict["id"]
            year = int(raw_dict["update_date"][:4])
            class_idx = _year2yidx(year)
            class_str = CLASS_STRS[class_idx]
            sample = DataSample(
                id=id,
                text=raw_dict["abstract"],
                domain_str=category,
                class_str=class_str,
                class_idx=class_idx,
            )
            collection.add_sample(sample)

        collection.class_strs = CLASS_STRS
        collection.domain_strs = ARXIV_CATEGORIES
        collection.populate_random_split()
        collection.populate_class_distribution()

        return collection


# example json line
# {
#     "id": "0704.0010",
#     "submitter": "Sergei Ovchinnikov",
#     "authors": "Sergei Ovchinnikov",
#     "title": "Partial cubes: structures, characterizations, and constructions",
#     "comments": "36 pages, 17 figures",
#     "journal-ref": null,
#     "doi": null,
#     "report-no": null,
#     "categories": "math.CO",
#     "license": null,
#     "abstract": "  Partial cubes are isometric subgraphs of hypercubes. Structures on a graph\ndefined by means of semicubes, and Djokovi\\'{c}'s and Winkler's relations play\nan important role in the theory of partial cubes. These structures are employed\nin the paper to characterize bipartite graphs and partial cubes of arbitrary\ndimension. New characterizations are established and new proofs of some known\nresults are given.\n  The operations of Cartesian product and pasting, and expansion and\ncontraction processes are utilized in the paper to construct new partial cubes\nfrom old ones. In particular, the isometric and lattice dimensions of finite\npartial cubes obtained by means of these operations are calculated.\n",
#     "versions": [
#         {
#             "version": "v1",
#             "created": "Sat, 31 Mar 2007 05:10:16 GMT"
#         }
#     ],
#     "update_date": "2007-05-23",
#     "authors_parsed": [
#         [
#             "Ovchinnikov",
#             "Sergei",
#             ""
#         ]
#     ]
# }
