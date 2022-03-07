from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from mda.data.data_collection import DataCollection, DataSample
from mda.registry import FromConfigBase, Registry, import_all


class MultiDomainDataset(FromConfigBase):
    def __init__(
        self,
        collection: DataCollection,
        use_domain_strs: Optional[List[str]],
    ) -> None:
        super().__init__()
        self.collection: DataCollection = collection
        assert all(
            domain_str in collection.domain_strs for domain_str in use_domain_strs
        )
        if use_domain_strs is None:
            self.filtered_samples: List[DataSample] = list(
                self.collection.samples.values()
            )
        else:
            self.filtered_samples: List[DataSample] = [
                sample
                for sample in collection.samples.values()
                if sample.domain_str in use_domain_strs
            ]
        self.use_domain_strs = use_domain_strs

    def get_loader(self) -> Iterable[Dict[str, torch.Tensor]]:
        raise NotImplementedError()


DATASET_REGISTRY = Registry(MultiDomainDataset)
import_all(Path(__file__).parent, "mda.data")
