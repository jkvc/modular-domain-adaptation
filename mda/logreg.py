import torch
from pydantic import BaseModel
from torch.optim import SGD
from tqdm import trange

from mda.data import MultiDomainDataset
from mda.model import Model
from mda.util import AUTO_DEVICE


def train_logreg_model(
    model: Model,
    dataset: MultiDomainDataset,
    num_epoch: int = 5000,
    learning_rate: float = 1e-1,
    num_dataloader_worker=6,
):
    optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=0)

    model.train()
    for e in trange(num_epoch):
        loader = dataset.get_loader(num_worker=num_dataloader_worker)
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            pred_batch = model(batch)
            loss = pred_batch["loss"]
            loss.backward()
            optimizer.step()


def eval_logreg_model(
    model: Model,
    dataset: MultiDomainDataset,
    num_dataloader_worker=6,
) -> float:
    loader = dataset.get_loader(num_worker=num_dataloader_worker)
    num_correct = 0
    num_samples = 0
    for batch in loader:
        num_samples += len(batch["bow"])
        pred_batch = model(batch)
        pred = torch.argmax(pred_batch["logits"], dim=-1)
        is_correct = pred == batch["class_idx"]
        num_correct += is_correct.sum()
    return (num_correct / num_samples).item()
