import random
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel


class DataSample(BaseModel):
    id: str
    text: str
    domain_str: str
    domain_idx: int
    class_str: Optional[str]
    class_idx: Optional[int]


class ClassDistribution(BaseModel):
    train_domain2prop: Dict[str, List[float]] = {}
    test_domain2prop: Dict[str, List[float]] = {}


class DataCollection(BaseModel):
    class_strs: List[str] = []
    domain_strs: List[str] = []
    class_dist: ClassDistribution = ClassDistribution()
    samples: Dict[str, DataSample] = {}
    train_ids: List[str] = []
    test_ids: List[str] = []

    def add_sample(self, sample: DataSample) -> None:
        assert sample.id not in self.samples
        self.samples[sample.id] = sample

    def populate_train_test_split_ids(
        self,
        random_seed: Optional[int] = None,
        train_prop: float = 0.8,
    ) -> None:
        all_ids = sorted(list(self.samples.keys()))
        rng = random.Random(seed=random_seed)
        rng.shuffle(all_ids)

        n_train_samples = int(len(all_ids) * train_prop)

        self.train_ids = all_ids[:n_train_samples]
        self.test_ids = all_ids[n_train_samples:]

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
            return {
                domain: (count / count.sum()).tolist()
                for domain, count in domain2count.items()
            }

        self.class_dist.train_domain2prop = calculate_class_distribution(
            [self.samples[id] for id in self.train_ids]
        )
        self.class_dist.test_domain2prop = calculate_class_distribution(
            [self.samples[id] for id in self.test_ids]
        )


d = DataCollection()
print(d.class_dist)
