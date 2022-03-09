from mda.data import MultiDomainDataset
from mda.logreg import train_logreg_model
from mda.model import Model
from mda.registry import FromConfigBase, Registry


class Trainer(FromConfigBase):
    def __init__(self, model: Model, dataset: MultiDomainDataset):
        self.model = model
        self.dataset = dataset

    def run(self):
        raise NotImplementedError()


TRAINER_REGISTRY = Registry(Trainer)


@TRAINER_REGISTRY.register("logreg")
class LogregTrainer(Trainer):
    def __init__(self, model: Model, dataset: MultiDomainDataset):
        super().__init__(model, dataset)

    def run(self):
        train_logreg_model(self.model, self.dataset)
