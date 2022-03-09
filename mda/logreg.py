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
