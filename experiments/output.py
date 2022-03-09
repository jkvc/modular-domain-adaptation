import logging
from pathlib import Path

import pandas as pd
import torch
from mda.data import MultiDomainDataset
from mda.logreg import eval_logreg_model
from mda.model import Model
from mda.model.logreg import LogisticRegressionModel
from mda.registry import FromConfigBase, Registry, import_all
from mda.util import save_json

from experiments.acc import ModelAccuracy

logger = logging.getLogger(__name__)


class Output(FromConfigBase):
    def __init__(
        self,
        train_dataset: MultiDomainDataset,
        test_dataset: MultiDomainDataset,
        model: Model,
        output_dir: str,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.output_dir = output_dir

    def execute(self):
        raise NotImplementedError


OUTPUT_REGISTRY = Registry(Output)


@OUTPUT_REGISTRY.register("acc")
class AccOutput(Output):
    def _eval(
        self,
        model: Model,
        dataset: MultiDomainDataset,
        num_dataloader_worker=6,
    ) -> float:
        model.eval()
        loader = dataset.get_loader(num_worker=num_dataloader_worker)
        num_correct = 0
        num_samples = 0
        for batch in loader:
            num_samples += len(batch["bow"])
            with torch.no_grad():
                pred_batch = model(batch)
            pred = torch.argmax(pred_batch["logits"], dim=-1)
            is_correct = pred == batch["class_idx"]
            num_correct += is_correct.sum()
        return (num_correct / num_samples).item()

    def execute(self):
        train_acc = self._eval(self.model, self.train_dataset)
        test_acc = self._eval(self.model, self.test_dataset)
        logger.info(f"train_acc={train_acc} test_acc={test_acc}")
        acc = ModelAccuracy(train_acc=train_acc, test_acc=test_acc)
        save_json(acc.dict(), f"{self.output_dir}/acc.json")


@OUTPUT_REGISTRY.register("lexicon")
class LexiconOutput(Output):
    def execute(self):
        lexicon_pd: pd.DataFrame = self.model.get_weighted_lexicon(
            vocab=self.train_dataset.vocab,
            colnames=self.train_dataset.collection.class_strs,
        )
        lexicon_pd.to_csv(f"{self.output_dir}/lexicon.csv")