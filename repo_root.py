import os
from pathlib import Path


def get_full_path(rel_path: str) -> str:
    return os.path.join(Path(__file__).parent, rel_path)
