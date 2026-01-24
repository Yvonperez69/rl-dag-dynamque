
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def encode_obs(obs: Dict[str, Any]) -> np.ndarray:
    return np.array([
        len(obs["graph"]),                 # taille du graphe
        obs["steps"] / obs["max_steps"],   # progression normalisée
        1.0 if obs["done"] else 0.0,        # terminal ou pas
    ], dtype=np.float32)

