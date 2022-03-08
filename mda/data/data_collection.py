import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel


class DataSample(BaseModel):
    id: str
    text: str
    domain_str: str
    class_str: Optional[str]
    class_idx: Optional[int]


class DataCollection(BaseModel):
    class_strs: List[str] = []
    domain_strs: List[str] = []
    class_dist: Dict[
        str, List[float]
    ] = {}  # dict maps a domain name to a list summing to 1
    samples: Dict[str, DataSample] = {}

    def add_sample(self, sample: DataSample) -> None:
        assert sample.id not in self.samples
        self.samples[sample.id] = sample

    def populate_class_distribution(self) -> None:
        domain2count = {
            domain_str: (np.zeros((len(self.class_strs),)) + 1e-8)
            for domain_str in self.domain_strs
        }
        for s in self.samples.values():
            domain2count[s.domain_str][s.class_idx] += 1
        domain2prop = {
            domain: (count / count.sum()).tolist()
            for domain, count in domain2count.items()
        }
        self.class_dist = domain2prop


def create_random_split(
    ids: List[str],
    train_prop: float = 0.8,
) -> Tuple[List[str], List[str]]:
    """
    set seed before calling this to ensure reproducibility
    """
    random.shuffle(ids)
    n_train_samples = int(len(ids) * train_prop)
    return ids[:n_train_samples], ids[n_train_samples:]
