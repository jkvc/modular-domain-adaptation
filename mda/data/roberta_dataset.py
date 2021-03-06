import logging
import re
from collections import Counter
from typing import Dict, Iterable, List, Literal, Optional

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizerFast

from ..util import AUTO_DEVICE
from . import DATASET_REGISTRY, MultiDomainDataset
from .data_collection import DataCollection

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register("roberta_tokenize")
class RobertaTokenizeDataset(MultiDomainDataset, Dataset):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        collection: DataCollection,
        use_domain_strs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            batch_size,
            num_workers,
            collection,
            use_domain_strs,
        )

        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.domain_str2domain_idx = {
            domain_str: idx for idx, domain_str in enumerate(self.domain_strs)
        }

    def __len__(self):
        return len(self.filtered_samples)

    def __getitem__(self, idx):
        sample = self.filtered_samples[idx]
        roberta_tokens = np.array(
            self.tokenizer.encode(
                sample.text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
            )
        )

        return {
            "roberta_tokens": roberta_tokens,
            "class_idx": sample.class_idx if sample.class_idx is not None else -1,
            "domain_idx": self.domain_str2domain_idx[sample.domain_str],
            "class_distribution": torch.FloatTensor(
                self.class_distribution[sample.domain_str]
            ),
        }

    def get_loader(self, shuffle: bool = True) -> Iterable[Dict[str, torch.Tensor]]:
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
