
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dag_env import DAGEnv


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
            nn.linear(hidden,hidden),
            nn.ReLU(),
            nn.linear(hidden,1)
        )
        
        def forward(self, obs: torch.Tensor) -> torch.Tensor : 
            return self.network(obs)

class AgentPPO :
    
    def __init__(self, env: DAGEnv, lr: float, gae_lambda: float, eps_clip: float, entropy_coef: float) :
        
    #peut etre des hyper param à implementer env lr
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        
        # On prends les NNs
        self.policy_network = PolicyNetwork(env.obs_dim, env.act_dim).to(self.device)
        self.value_network = ValueNetwork(env.obs_dim).to(self.device)
        
        # Leur optimiseurs
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=lr)
        
        
        def select_action(self, state: int, training: bool):
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                
                action_logits = self.policy_network(state_tensor)
                
                action_dist = torch.distributions.Categorical(logits=action_logits) # distribution des actions possibles, c'est la loi en gros
                
                state_value = self.value_network(state_tensor)
                
                if training :
                    action = action_dist.sample()
                else:
                    action = torch.argmax(action_dist.probs) #action_dist.probs renvoie la proba de chaque action
                
                log_prob = action_dist.log_prob(action)
                
                # .item() sert à convertir un tensor 0-D en float
                return action.item(), state_value.item(), log_prob.item() 
            
        def store_transition(self, state, action, reward, value, log_prob, done):

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_prob.append(log_prob)
            self.dones.append(done)
                    
        def update_policy(self, epochs: int, batch_size: int = 64):
            