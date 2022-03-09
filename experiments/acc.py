from __future__ import annotations

from os import listdir
from os.path import isdir
from typing import Dict, List, Optional, Union

from mda.util import get_full_path, load_json, save_json
from pydantic import BaseModel


class ModelAccuracy(BaseModel):
    train_acc: float
    test_acc: float


class RecursiveAccuracy(BaseModel):
    mean_acc: float
    children_rank: Optional[List[str]]
    children: Optional[Dict[str, Union[RecursiveAccuracy, float]]]


def collect_and_reduce_acc(root_dir: str):
    files = listdir(root_dir)
    is_leaf = "acc.json" in files
    print(files, is_leaf)

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
        # sort children desc by acc
        train_acc.children_rank = [
            child_name
            for child_name, child_acc in sorted(
                [(k, v) for k, v in train_acc.children.items()],
                reverse=True,
                key=lambda x: x[1].mean_acc
                if isinstance(x[1], RecursiveAccuracy)
                else x[1],
            )
        ]
        test_acc.children_rank = [
            child_name
            for child_name, child_acc in sorted(
                [(k, v) for k, v in test_acc.children.items()],
                reverse=True,
                key=lambda x: x[1].mean_acc
                if isinstance(x[1], RecursiveAccuracy)
                else x[1],
            )
        ]
        return train_acc, test_acc


if __name__ == "__main__":
    train_acc, test_acc = collect_and_reduce_acc(get_full_path("wkdir"))
    save_json(train_acc.dict(exclude_unset=True), get_full_path("wkdir/train_acc.json"))
    save_json(test_acc.dict(exclude_unset=True), get_full_path("wkdir/test_acc.json"))