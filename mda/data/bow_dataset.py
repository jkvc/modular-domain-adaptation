import logging
import re
from collections import Counter
from typing import Dict, Iterable, List, Literal, Optional

import nltk
import numpy as np
import torch
from mda.data.data_collection import DataCollection
from mda.util import AUTO_DEVICE
from nltk.corpus import stopwords

from . import DATASET_REGISTRY, MultiDomainDataset

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register("bow_single_batch")
class BagOfWordsSingleBatchDataset(MultiDomainDataset):
    def __init__(
        self,
        collection: DataCollection,
        use_domain_strs: Optional[List[str]] = None,
        # specify exaclty one of following two
        vocab_size: Optional[int] = None,
        vocab_override: Optional[List[str]] = None,
        # specify exactly one of following two
        class_distribution_use_split: Optional[Literal["train", "test"]] = None,
        class_distribution_override: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        super().__init__(collection, use_domain_strs)
        self.all_sample_tokens = self.get_all_sample_tokens()

        assert (vocab_override is not None) != (vocab_size is not None)
        if vocab_override:
            self.vocab: List[str] = vocab_override
        else:
            self.vocab = self.build_vocab(vocab_size)

        assert (class_distribution_use_split is not None) != (
            class_distribution_override is not None
        )
        if class_distribution_override:
            self.class_distribution = class_distribution_override
        else:
            if class_distribution_use_split == "train":
                self.class_distribution = collection.class_dist.train_class_dist
            elif class_distribution_use_split == "test":
                self.class_distribution = collection.class_dist.test_class_dist
            else:
                raise ValueError()

        self.batch = self.build_batch()

    def get_all_sample_tokens(self):
        try:
            sws = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            sws = set(stopwords.words("english"))

        all_sample_tokens: List[List[str]] = []
        for sample in self.filtered_samples:
            text = sample.text
            nopunc = re.sub(r"[^\w\s]", "", text)
            tokens = nopunc.split()
            tokens = [
                w
                for w in tokens
                if (not w.startswith("@") and w not in sws and not w.isdigit())
            ]
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
        return batch

    def get_loader(self, num_worker: int) -> Iterable[Dict[str, torch.Tensor]]:
        return _single_batch_iterator(self.batch)


class _single_batch_iterator:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        self.returned = False
        return self

    def __next__(self):
        if not self.returned:
            self.returned = True
            return self.batch
        else:
            raise StopIteration
