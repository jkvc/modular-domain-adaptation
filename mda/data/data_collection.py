import random
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel


class DataSample(BaseModel):
    id: str
    text: str
    domain_str: str
    class_str: Optional[str]
    class_idx: Optional[int]


class DataSplit(BaseModel):
    train_ids: List[str] = []
    test_ids: List[str] = []


class ClassDistribution(BaseModel):
    # dict maps a domain name to a list summing to 1
    train_class_dist: Dict[str, List[float]] = {}
    test_class_dist: Dict[str, List[float]] = {}


class DataCollection(BaseModel):
    class_strs: List[str] = []
    domain_strs: List[str] = []
    class_dist: ClassDistribution = ClassDistribution()
    samples: Dict[str, DataSample] = {}
    split: DataSplit = DataSplit()

    def add_sample(self, sample: DataSample) -> None:
        assert sample.id not in self.samples
        self.samples[sample.id] = sample

    def populate_random_split(
        self,
        train_prop: float = 0.8,
    ) -> None:
        """
        set seed before calling this to ensure reproducibility
        """
        all_ids = sorted(list(self.samples.keys()))
        random.shuffle(all_ids)

        n_train_samples = int(len(all_ids) * train_prop)

        self.split.train_ids = all_ids[:n_train_samples]
        self.split.test_ids = all_ids[n_train_samples:]

    def populate_class_distribution(self) -> None:
        def calculate_class_distribution(
            samples: List[DataSample],
        ) -> Dict[str, List[float]]:
            domain2count = {
                domain_str: (np.zeros((len(self.class_strs),)) + 1e-8)
                for domain_str in self.domain_strs
            }
            for s in samples:
                domain2count[s.domain_str][s.class_idx] += 1
            domain2prop = {
                domain: (count / count.sum()).tolist()
                for domain, count in domain2count.items()
            }
            return domain2prop

        self.class_dist.train_class_dist = calculate_class_distribution(
            [self.samples[id] for id in self.split.train_ids]
        )
        self.class_dist.test_class_dist = calculate_class_distribution(
            [self.samples[id] for id in self.split.test_ids]
        )
