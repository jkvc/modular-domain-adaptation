import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from mda.data import MultiDomainDataset
from mda.model.common import ReversalLayer

from . import MODEL_REGISTRY, Model

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("logreg")
class LogisticRegressionModel(Model):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        n_domains: Optional[int] = None,
        use_direct_residualization: bool = False,
        use_gradient_reversal: bool = False,
        use_domain_specific_bias: bool = False,
        use_domain_specific_normalization: bool = False,
        hidden_size: int = 64,
        regularization_constant: float = 1e-4,
    ):
        super().__init__()

        if (
            use_domain_specific_bias
            or use_gradient_reversal
            or use_direct_residualization
        ):
            assert n_domains is not None
        else:
            n_domains = 1

        self.vocab_size: int = vocab_size
        self.n_classes: int = n_classes
        self.n_domains: int = n_domains
        self.use_direct_residualization: bool = use_direct_residualization
        self.use_gradient_reversal: bool = use_gradient_reversal
        self.use_domain_specific_bias: bool = use_domain_specific_bias
        self.use_domain_specific_normalization: bool = use_domain_specific_normalization
        self.hidden_size: int = hidden_size
        self.regularization_constant: float = regularization_constant

        self.tff = nn.Linear(self.vocab_size, self.hidden_size, bias=False)
        self.yout = nn.Linear(self.hidden_size, self.n_classes, bias=False)
        self.cff = nn.Sequential(
            nn.Linear(self.n_domains, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.n_classes),
        )
        self.cout = nn.Sequential(
            ReversalLayer(),
            nn.Linear(self.hidden_size, self.n_domains),
        )

    def forward(
        self, batch: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, torch.Tensor]:
        assert "class_strs" in batch
        assert "vocab" in batch
        self.class_strs = batch["class_strs"]
        self.vocab = batch["vocab"]

        bow = batch["bow"].to(torch.float)
        batchsize, vocabsize = bow.shape
        assert vocabsize == self.vocab_size

        # normalize word frequency at each domain
        # this requires `batch` is the entire dataset
        if self.use_domain_specific_normalization:

            def normalize_batch(bow_batch):
                for domain_idx in range(self.n_domains):
                    sample_idxs = torch.where(batch["domain_idx"] == domain_idx)
                    if len(sample_idxs) == 0:
                        continue
                    bow_batch[sample_idxs] -= bow_batch[sample_idxs].mean(axis=0)
                return bow_batch

            if not self.training:
                bow = normalize_batch(bow)
            elif not hasattr(self, "bow_full_batch_normed"):
                self.bow_full_batch = bow
                self.bow_full_batch_normed = normalize_batch(bow)
                bow = self.bow_full_batch_normed
            else:
                assert (self.bow_full_batch == bow).all()
                bow = self.bow_full_batch_normed

        e = self.tff(bow)
        logits = self.yout(e)

        if self.use_domain_specific_bias:
            class_distribution = batch["class_distribution"].to(torch.float)
            logits = logits + torch.log(class_distribution)

        if self.use_direct_residualization:
            if self.training:
                domain_onehot = (
                    torch.eye(self.n_domains)[batch["domain_idx"]]
                    .to(torch.float)
                    .to(bow.device)
                )
                class_pred_logits = self.cff(domain_onehot)
                logits = logits + class_pred_logits

        batch["logits"] = logits

        if self.training:
            class_idx = batch["class_idx"]
            loss = nnf.cross_entropy(logits, class_idx, reduction="none")

            if self.use_gradient_reversal:
                if self.training:
                    confound_logits = self.cout(e)
                    domain_idx = batch["domain_idx"]
                    confound_loss = nnf.cross_entropy(
                        confound_logits, domain_idx, reduction="none"
                    )
                    loss = loss + confound_loss

            # L1 regularization on t weights only
            loss = (
                loss
                + torch.abs(self.yout.weight @ self.tff.weight).sum()
                * self.regularization_constant
            )
            loss = loss.mean()

            batch["loss"] = loss

        return batch

    def get_weighted_lexicon(
        self, vocab: List[str], colnames: List[str]
    ) -> pd.DataFrame:
        weights = (
            self.yout.weight.data.detach().cpu().numpy()
            @ self.tff.weight.data.detach().cpu().numpy()
        )
        return elicit_lexicon(weights, vocab, colnames)

    def to_logdir(self, logdir: str):
        lexicon_pd: pd.DataFrame = self.get_weighted_lexicon(
            vocab=self.vocab,
            colnames=self.class_strs,
        )
        lexicon_pd.to_csv(f"{logdir}/lexicon.csv")

    def from_logdir(self, log_dir: str):
        self.single_matrix_model = LogisticRegressionSingleWeightMatrixModel(
            vocab_size=self.vocab_size,
            n_classes=self.n_classes,
            n_domains=self.n_domains,
            use_domain_specific_bias=self.use_domain_specific_bias,
            use_domain_specific_normalization=self.use_domain_specific_normalization,
        ).to(self.tff.weight.device)
        df = pd.read_csv(f"{log_dir}/lexicon.csv", index_col=0)
        self.single_matrix_model.load_lexicon(df)
        logger.info(
            f"loaded weights from lexicon csv, vocab_size {len(df)} n_classes {len(df.columns)-1}"
        )
        self.train = self.single_matrix_model.train
        self.forward = self.single_matrix_model.forward


class LogisticRegressionSingleWeightMatrixModel(Model):
    """
    Similar to `LogisticRegressionModel`, except the singular weight matrix
    This is useful for evaluating an existing lexicon
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        n_domains: Optional[int] = None,
        use_domain_specific_bias: bool = False,
        use_domain_specific_normalization: bool = False,
    ):
        super().__init__()

        if use_domain_specific_bias:
            assert n_domains is not None
        else:
            n_domains = 1

        self.vocab_size: int = vocab_size
        self.n_classes: int = n_classes
        self.n_domains: int = n_domains
        self.use_domain_specific_bias: bool = use_domain_specific_bias
        self.use_domain_specific_normalization: bool = use_domain_specific_normalization
        self.ff = nn.Linear(self.vocab_size, self.n_classes, bias=False)

        self.eval()

    def train(self, is_train=True):
        if is_train:
            raise NotImplementedError(
                "LogisticRegressionSingleWeightMatrixModel does not support train()"
            )
        super().train(False)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert not self.training

        bow = batch["bow"].to(torch.float)
        batchsize, vocabsize = bow.shape
        assert vocabsize == self.vocab_size

        # test time only, no cacheing, slower
        if self.use_domain_specific_normalization:

            def normalize_batch(bow_batch):
                for domain_idx in range(self.n_domains):
                    sample_idxs = torch.where(batch["domain_idx"] == domain_idx)
                    if len(sample_idxs) == 0:
                        continue
                    bow_batch[sample_idxs] -= bow_batch[sample_idxs].mean(axis=0)
                return bow_batch

            bow = normalize_batch(bow)

        logits = self.ff(bow)

        if self.use_domain_specific_bias:
            class_distribution = batch["class_distribution"].to(torch.float)
            logits = logits + torch.log(class_distribution)

        batch["logits"] = logits
        return batch

    def load_lexicon(self, df: pd.DataFrame):
        class_strs = [colname for colname in df.columns if colname != "word"]
        n_classes = len(class_strs)
        vocab_size = len(df)

        weight_matrix = np.zeros((n_classes, vocab_size))
        for class_idx, class_str in enumerate(class_strs):
            weight_matrix[class_idx] = df[class_str]

        with torch.no_grad():
            self.ff.weight.copy_(torch.FloatTensor(weight_matrix))
            self.ff.weight.requires_grad = False


def elicit_lexicon(
    weights: np.ndarray, vocab: List[str], colnames: List[str]
) -> pd.DataFrame:
    nclass, vocabsize = weights.shape
    assert len(vocab) == vocabsize
    assert len(colnames) == nclass

    df = pd.DataFrame()
    df["word"] = vocab
    for c in range(nclass):
        df[colnames[c]] = weights[c]
    return df
