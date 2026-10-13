"""Global seeding so runs are repeatable."""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python `random`, numpy, torch (CPU + CUDA); set cudnn deterministic flags."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # preprocessing-only environments
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
