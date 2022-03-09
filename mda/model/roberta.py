from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from mda.model.common import ReversalLayer
from transformers import RobertaModel

from . import MODEL_REGISTRY, Model

ROBERAT_EMB_SIZE = 768


@MODEL_REGISTRY.register("roberta")
class RobertaClassifier(Model):
    def __init__(
        self,
        n_classes: int,
        n_domains: Optional[int] = None,
        dropout_p: float = 0.2,
        use_direct_residualization: bool = False,
        use_domain_specific_bias: bool = False,
        use_gradient_reversal: bool = False,
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

        self.n_classes: int = n_classes
        self.n_domains: Optional[int] = n_domains
        self.dropout_p: float = dropout_p
        self.use_direct_residualization: bool = use_direct_residualization
        self.use_domain_specific_bias: bool = use_domain_specific_bias
        self.use_gradient_reversal: bool = use_gradient_reversal

        self.roberta = RobertaModel.from_pretrained(
            "roberta-base", hidden_dropout_prob=self.dropout_p
        )

        self.yff = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(ROBERAT_EMB_SIZE, ROBERAT_EMB_SIZE),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(ROBERAT_EMB_SIZE, self.n_classes),
        )

        if use_direct_residualization:
            self.cff = nn.Sequential(
                nn.Linear(self.n_domains, 64),
                nn.Tanh(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(64, self.n_classes),
            )

        self.cout = nn.Sequential(
            ReversalLayer(),
            nn.Linear(ROBERAT_EMB_SIZE, self.n_sources),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["roberta_tokens"]
        x = self.roberta(x)[0]
        # huggingface robertaclassifier applies dropout before this, we apply dropout after this
        # shouldnt make a big difference
        e = x[:, 0, :]  # the <s> tokens, i.e. <CLS>
        logits = self.yff(e)

        if self.use_domain_specific_bias:
            labelprops = batch["class_distribution"].to(torch.float)  # nsample, nclass
            logits = logits + torch.log(labelprops)

        if self.use_direct_residualization:
            if self.training:
                domain_onehot = (
                    torch.eye(self.n_domains)[batch["domain_idx"]]
                    .to(torch.float)
                    .to(x.device)
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
        loss = loss.mean()
        batch.update(
            {
                "logits": logits,
                "loss": loss,
            }
        )
        return batch
