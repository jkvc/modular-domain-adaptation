from tqdm import tqdm
from transformers import AdamW

from mda.data import MultiDomainDataset
from mda.model import Model
from mda.util import AUTO_DEVICE


def train_roberta_model(
    model: Model,
    dataset: MultiDomainDataset,
    num_epoch: int = 5,
    learning_rate: float = 1e-5,
):
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loader = dataset.get_loader()

    for e in range(num_epoch):
        for batch in tqdm(loader, desc=f"epoch {e}"):
            batch = {k: v.to(AUTO_DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            pred_batch = model(batch)
            loss = pred_batch["loss"]
            loss.backward()
            optimizer.step()
            return  # fixme
