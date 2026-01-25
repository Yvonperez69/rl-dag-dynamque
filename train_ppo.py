
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


class PolicyNetwork(nn.Module):
    
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        
        # Encodeur
        self.encodeur = nn.Sequential(
            nn.Linear(obs_dim,hidden),
            nn.ReLU(),
            nn.Sequential(hidden,hidden),
            nn.ReLU()
        )
        
        self.policy_head = nn.linear(hidden,act_dim)
        
        def forward(self, obs: torch.Tensor) -> torch.Tensor :
            encoded = self.encodeur(obs)
            action_logits = self.policy_head(encoded)
            return action_logits
        
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.linear(obs_dim,hidden),
            nn.ReLU(),
            nn.linear(hidden,hidden)
            nn.ReLU()
            nn.linear(hidden,1)
        )
        
        def forward(self, obs: torch.Tensor) -> torch.Tensor : 
            return self.network(obs)

class AgentPPO :
    v