from __future__ import annotations

import logging
import sys
from os import listdir
from os.path import isdir
from typing import Dict, List, Optional, Union

import torch
from mda.data import MultiDomainDataset
from mda.model import Model
from mda.util import AUTO_DEVICE, get_full_path, load_json, save_json
from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelAccuracy(BaseModel):
    train_acc: float
    test_acc: float


def compute_accs(
    model: Model,
    train_dataset: Optional[MultiDomainDataset],
    test_dataset: Optional[MultiDomainDataset],
) -> ModelAccuracy:
    def compute_acc(model, dataset) -> float:
        model.eval()
        loader = dataset.get_loader()
        num_correct = 0
        num_samples = 0
        for batch in tqdm(loader):
            num_samples += len(batch["class_idx"])
            with torch.no_grad():
                batch = {
                    k: (v.to(AUTO_DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                pred_batch = model(batch)
            pred = torch.argmax(pred_batch["logits"], dim=-1)
            is_correct = pred == batch["class_idx"]
            num_correct += is_correct.sum()
        return (num_correct / num_samples).item()

    train_acc = compute_acc(model, train_dataset) if train_dataset else 0
    test_acc = compute_acc(model, test_dataset) if test_dataset else 0
    logger.info(f"train_acc={train_acc} test_acc={test_acc}")
    acc = ModelAccuracy(train_acc=train_acc, test_acc=test_acc)
    return acc


class RecursiveAccuracy(BaseModel):
    mean_acc: float
    children_reduced: Optional[Dict[str, float]]
    children: Optional[Dict[str, Union[RecursiveAccuracy, float]]]


def collect_and_reduce_acc(root_dir: str):
    files = sorted(listdir(root_dir))
    is_leaf = "acc.json" in files

    if is_leaf:
        model_acc = ModelAccuracy.parse_obj(load_json(f"{root_dir}/acc.json"))
        return model_acc.train_acc, model_acc.test_acc
    else:
        train_acc = RecursiveAccuracy(mean_acc=0, children={})
        test_acc = RecursiveAccuracy(mean_acc=0, children={})
        for child_dir in files:
            if not isdir(f"{root_dir}/{child_dir}"):
                continue
            child_train_acc, child_test_acc = collect_and_reduce_acc(
                f"{root_dir}/{child_dir}"
            )
            train_acc.children[child_dir] = child_train_acc
            test_acc.children[child_dir] = child_test_acc

        # calc mean
        train_acc.mean_acc = sum(
            child_train_acc.mean_acc
            if isinstance(child_train_acc, RecursiveAccuracy)
            else child_train_acc
            for child_train_acc in train_acc.children.values()
        ) / len(train_acc.children)
        test_acc.mean_acc = sum(
            child_test_acc.mean_acc
            if isinstance(child_test_acc, RecursiveAccuracy)
            else child_test_acc
            for child_test_acc in test_acc.children.values()
        ) / len(test_acc.children)
        # calculate reduced child mean for easier read
        train_acc.children_reduced = {
            child_name: child.mean_acc
            if isinstance(child, RecursiveAccuracy)
            else child
            for child_name, child in train_acc.children.items()
        }
        test_acc.children_reduced = {
            child_name: child.mean_acc
            if isinstance(child, RecursiveAccuracy)
            else child
            for child_name, child in test_acc.children.items()
        }
        return train_acc, test_acc


if __name__ == "__main__":
    train_acc, test_acc = collect_and_reduce_acc(get_full_path(f"{sys.argv[1]}"))
    save_json(
        train_acc.dict(exclude_unset=True),
        get_full_path(f"{sys.argv[1]}/train_acc.json"),
    )
    save_json(
        test_acc.dict(exclude_unset=True), get_full_path(f"{sys.argv[1]}/test_acc.json")
    )
