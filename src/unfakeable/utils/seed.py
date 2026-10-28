import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (if importable) for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
