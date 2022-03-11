import logging
import re
from collections import Counter
from typing import Dict, Iterable, List, Literal, Optional

import nltk
import numpy as np
import pandas as pd
import torch
from genericpath import exists
from mda.data.data_collection import DataCollection
from mda.util import AUTO_DEVICE
from nltk.corpus import stopwords

from . import DATASET_REGISTRY, MultiDomainDataset

logger = logging.getLogger(__name__)

try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    _STOPWORDS = set(stopwords.words("english"))


def get_vocab_from_lexicon_csv(log_dir) -> Optional[List[str]]:
    if not exists(f"{log_dir}/lexicon.csv"):
        return None
    df = pd.read_csv(f"{log_dir}/lexicon.csv")
    return df["word"].tolist()


def tokenize(text: str) -> List[str]:
    nopunc = re.sub(r"[^\w\s]", "", text)
    tokens = nopunc.split()
    tokens = [
        w
        for w in tokens
        if (not w.startswith("@") and w not in _STOPWORDS and not w.isdigit())
    ]
    return tokens


@DATASET_REGISTRY.register("bow_single_batch")
class BagOfWordsSingleBatchDataset(MultiDomainDataset):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        collection: DataCollection,
        use_domain_strs: Optional[List[str]] = None,
        vocab_size: Optional[int] = None,
        vocab_override: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            batch_size,
            num_workers,
            collection,
            use_domain_strs,
        )
        self.all_sample_tokens = self.get_all_sample_tokens()

        if vocab_override:
            logger.info("using vocab override")
            self.vocab: List[str] = vocab_override
        else:
            logger.info(f"building vocab of size {vocab_size}")
            self.vocab = self.build_vocab(vocab_size)

        self.batch = self.build_batch()

    def get_all_sample_tokens(self):
        all_sample_tokens: List[List[str]] = []
        for sample in self.filtered_samples:
            text = sample.text.lower()
            tokens = tokenize(text)
            all_sample_tokens.append(tokens)
        return all_sample_tokens

    def build_vocab(self, vocab_size: int):
        word2count = Counter()
        for tokens in self.all_sample_tokens:
            word2count.update(tokens)
        vocab = [word for word, _ in word2count.most_common(vocab_size)]
        return vocab

    def build_batch(self) -> Dict[str, torch.Tensor]:
        word2idx = {w: i for i, w in enumerate(self.vocab)}

        bow = np.zeros((len(self.filtered_samples), len(word2idx)))

        class_idx = np.zeros((len(self.filtered_samples),))
        for i, sample in enumerate((self.filtered_samples)):
            tokens = self.all_sample_tokens[i]
            for w in tokens:
                if w in word2idx:
                    bow[i, word2idx[w]] = 1
            class_idx[i] = sample.class_idx

        domain_str2domain_idx = {
            domain_str: idx for idx, domain_str in enumerate(self.domain_strs)
        }
        domain_idx = torch.LongTensor(
            [domain_str2domain_idx[s.domain_str] for s in self.filtered_samples]
        )

        class_distribution = torch.FloatTensor(
            [self.class_distribution[s.domain_str] for s in self.filtered_samples]
        )

        batch = {
            "bow": torch.FloatTensor(bow),
            "class_distribution": class_distribution,
            "class_idx": torch.LongTensor(class_idx),
            "domain_idx": domain_idx.to(torch.long),
        }
        batch = {k: v.to(AUTO_DEVICE) for k, v in batch.items()}
        batch.update(
            {
                "class_strs": self.collection.class_strs,
                "vocab": self.vocab,
            }
        )
        return batch

    def get_loader(
        self,
    ) -> Iterable[Dict[str, torch.Tensor]]:
        assert self.batch_size == -1
        return _single_batch_iterator(self.batch)

    def computed_asset(self) -> Dict:
        return {"vocab_override": self.vocab}


class _single_batch_iterator:
    def __init__(self, batch):
        self.batch = batch
        self.returned = False

    def __iter__(self):
        self.returned = False
        return self

    def __next__(self):
        if not self.returned:
            self.returned = True
            return self.batch
        else:
            raise StopIteration
