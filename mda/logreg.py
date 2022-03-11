from typing import List

import numpy as np
import pandas as pd
import torch
from torch.optim import SGD
from tqdm import trange

from mda.data import MultiDomainDataset
from mda.data.bow_dataset import BagOfWordsSingleBatchDataset, tokenize
from mda.model import Model
from mda.model.logreg import (
    LogisticRegressionModel,
    LogisticRegressionSingleWeightMatrixModel,
)
from mda.util import AUTO_DEVICE


def train_logreg_model(
    model: Model,
    dataset: MultiDomainDataset,
    num_epoch: int = 5000,
    learning_rate: float = 1e-1,
):
    optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=0)

    model.train()
    for e in trange(num_epoch):
        loader = dataset.get_loader()
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            pred_batch = model(batch)
            loss = pred_batch["loss"]
            loss.backward()
            optimizer.step()


def train_lexicon(
    dataset: BagOfWordsSingleBatchDataset,
    use_domain_specific_bias: bool = True,
    use_domain_specific_normalization: bool = True,
    regularization_constant: float = 1e-4,
    num_epoch: int = 5000,
    learning_rate: float = 1e-1,
) -> pd.DataFrame:
    model = LogisticRegressionModel(
        vocab_size=len(dataset.vocab),
        n_classes=len(dataset.collection.class_strs),
        n_domains=len(dataset.domain_strs),
        use_domain_specific_bias=use_domain_specific_bias,
        use_domain_specific_normalization=use_domain_specific_normalization,
        regularization_constant=regularization_constant,
    )
    model = model.to(AUTO_DEVICE)
    train_logreg_model(
        model=model,
        dataset=dataset,
        num_epoch=num_epoch,
        learning_rate=learning_rate,
    )
    lexicon_df = model.get_weighted_lexicon(
        dataset.vocab, dataset.collection.class_strs
    )
    return lexicon_df


def lexicon_predict(
    lexicon_df: pd.DataFrame,
    dataset: BagOfWordsSingleBatchDataset,
    use_domain_specific_bias: bool = True,
    use_domain_specific_normalization: bool = True,
) -> torch.Tensor:
    model = LogisticRegressionSingleWeightMatrixModel(
        vocab_size=len(dataset.vocab),
        n_classes=len(dataset.collection.class_strs),
        n_domains=len(dataset.domain_strs),
        use_domain_specific_bias=use_domain_specific_bias,
        use_domain_specific_normalization=use_domain_specific_normalization,
    )
    model.load_lexicon(lexicon_df)
    model.to(AUTO_DEVICE)
    probs = []
    for batch in dataset.get_loader():
        with torch.no_grad():
            batch = model(batch)
            probs.append(torch.sigmoid(batch["logits"]))
    probs = torch.cat(probs, dim=0)
    return probs
