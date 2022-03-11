import pandas as pd
import torch
from tqdm import tqdm

from .data import MultiDomainDataset
from .data.bow_dataset import BagOfWordsSingleBatchDataset
from .logreg import train_logreg_model
from .model import Model
from .model.logreg import (
    LogisticRegressionModel,
    LogisticRegressionSingleWeightMatrixModel,
)
from .util import AUTO_DEVICE


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
    return predict(model, dataset)


def predict(model: Model, dataset: MultiDomainDataset) -> torch.Tensor:
    model.to(AUTO_DEVICE)
    probs = []
    for batch in tqdm(dataset.get_loader(shuffle=False)):
        with torch.no_grad():
            batch = {k: v.to(AUTO_DEVICE) for k, v in batch.items()}
            batch = model(batch)
            probs.append(torch.sigmoid(batch["logits"]))
    probs = torch.cat(probs, dim=0)
    return probs
