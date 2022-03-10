import json
import pickle
import random
import shutil
from multiprocessing import Pool, cpu_count
from os import makedirs, mkdir
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from genericpath import exists
from tqdm import tqdm

AUTO_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def read_txt_as_str_list(filepath: str) -> List[str]:
    with open(filepath) as f:
        ls = f.readlines()
    ls = [l.strip() for l in ls]
    return ls


def write_str_list_as_txt(lst: List[str], filepath: str):
    with open(filepath, "w") as f:
        f.writelines([f"{s}\n" for s in lst])


def save_pkl(obj, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(save_path: str):
    with open(save_path, "rb") as f:
        return pickle.load(f)


def save_json(obj, save_path: str):
    with open(save_path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(save_path: str):
    with open(save_path, "r") as f:
        return json.load(f)


def mkdirs(path: str, overwrite: bool = False):
    if exists(path):
        assert overwrite, f"{path} already exists"
        shutil.rmtree(path)
    makedirs(path)


def mark_experiment_done(path: str):
    write_str_list_as_txt(["done"], f"{path}/.is_done")


def is_experiment_done(path: str) -> bool:
    return exists(f"{path}/.is_done")


DEFAULT_FIGURE_DPI = 300


def save_plt(path: str, dpi: int = DEFAULT_FIGURE_DPI):
    # plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(path, dpi=dpi)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ParallelHandler:
    def __init__(self, f):
        self.f = f

    def f_wrapper(self, param):
        if isinstance(param, tuple) or isinstance(param, list):
            return self.f(*param)
        else:
            return self.f(param)

    def run(self, params, num_procs=(cpu_count()), desc=None, quiet=False):
        pool = Pool(
            processes=num_procs,
        )
        rets = list(
            tqdm(
                pool.imap_unordered(self.f_wrapper, params),
                total=len(params),
                desc=desc,
                disable=quiet,
            )
        )
        pool.close()
        return rets
