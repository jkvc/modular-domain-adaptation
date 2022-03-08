from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from mda.model.common import ReversalLayer

from . import MODEL_REGISTRY, Model


@MODEL_REGISTRY.register("logreg")
class LogisticRegressionModel(Model):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        n_domains: Optional[int],
        use_direct_residualization: bool = False,
        use_gradient_reversal: bool = False,
        use_domain_specific_bias: bool = False,
        use_domain_specific_normalization: bool = False,
        hidden_size: int = 64,
        regularization_constant: float = 1e-4,
    ):
        super().__init__()

        if use_domain_specific_bias or use_gradient_reversal:
            assert n_domains is not None
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bow = batch["bow"].to(torch.float)
        batchsize, vocabsize = bow.shape
        assert vocabsize == self.vocab_size

        # normalize word frequency at each domain
        # this works best if `batch` is the entire dataset
        if self.use_domain_specific_normalization:
            for domain_idx in range(self.n_domains):
                sample_idxs = torch.where(batch["domain_idx"] == domain_idx)
                if len(sample_idxs) == 0:
                    continue
                bow[sample_idxs] -= bow[sample_idxs].mean(axis=0)

        e = self.tff(bow)
        logits = self.yout(e)

        if self.use_domain_specific_bias:
            class_distribution = batch["class_distribution"].to(torch.float)
            logits = logits + torch.log(class_distribution)

        if self.use_direct_residualization:
            if self.training:
                domain_onehot = torch.eye(self.n_sources)[batch["domain_idx"]].to(
                    torch.float
                )
                class_pred_logits = self.cff(domain_onehot)
                logits = logits + class_pred_logits

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

        batch.update(
            {
                "logits": logits,
                "loss": loss,
            }
        )
        return batch

    def get_weighted_lexicon(
        self, vocab: List[str], colnames: List[str]
    ) -> pd.DataFrame:
        weights = (
            self.yout.weight.data.detach().cpu().numpy()
            @ self.tff.weight.data.detach().cpu().numpy()
        )
        nclass, vocabsize = weights.shape
        assert vocabsize == len(vocab)
        assert len(colnames) == nclass

        df = pd.DataFrame()
        df["word"] = vocab
        for c in range(nclass):
            df[colnames[c]] = weights[c]
        return df
