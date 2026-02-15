
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import all_tasks
import random

from agent_env import AgentEnv

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(obs)
        return self.policy_head(encoded)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class AgentPPO:
    def __init__(self, env: AgentEnv, lr: float, gae_lambda: float, eps_clip: float, entropy_coef: float, gamma: float = 0.99,):
        
        # hyperparamètres
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef

        # réseaux
        self.policy_network = PolicyNetwork(env.obs_dim, env.act_dim).to(self.device)
        self.value_network = ValueNetwork(env.obs_dim).to(self.device)

        # optimiseurs
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        # buffers de trajectoire
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

    def select_action(self, state: np.ndarray, training: bool) -> Tuple[int, float, float]:
        
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            
            action_logits = self.policy_network(state_tensor)
            
            if not torch.isfinite(action_logits).all():
                action = torch.randint(0, action_logits.shape[-1], (1,), device=self.device)
                return action.item(), 0.0, 0.0
            
            action_dist = torch.distributions.Categorical(logits=action_logits)
            state_value = self.value_network(state_tensor)

            if training:
                action = action_dist.sample()
            else:
                action = torch.argmax(action_dist.probs)

            log_prob = action_dist.log_prob(action)

        return action.item(), state_value.item(), log_prob.item()

    def store_transition(self, state: np.ndarray, action: int, reward: float, value: float, log_prob: float, done: bool) -> None:
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], next_value: float = 0.0, gamma: float | None = None, gae_lambda: float | None = None,) -> Tuple[List[float], List[float]]:
        """
        Calcule les avantages et les retours avec Generalized Advantage Estimation.
        Retourne (advantages, returns).
        """
        gamma = self.gamma if gamma is None else gamma
        lam = self.gae_lambda if gae_lambda is None else gae_lambda

        advantages: List[float] = []
        gae = 0.0
        # on ajoute une valeur de bootstrap pour v_{t+1}
        values_ext = values + [next_value]

        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values_ext[:-1])]
        return advantages, returns

    def update_policy(self, epochs: int, batch_size: int = 64) -> None:
        
        if len(self.states) == 0:
            return
        
        # sécurités anti-NaN
        states_tensor = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        if not torch.isfinite(states_tensor).all():
            # on jette le batch corrompu pour éviter de propager des NaN
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.values.clear()
            self.log_probs.clear()
            self.dones.clear()
            return
        
        # bootstrap pour le dernier état si non terminal
        with torch.no_grad():
            if self.dones and not self.dones[-1]:
                last_state = torch.as_tensor(self.states[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
                next_value = self.value_network(last_state).item()
            else:
                next_value = 0.0
        
        # Calcul des avantages et returns
        
        advantages, returns = self.compute_gae(self.rewards, self.values, self.dones, next_value=next_value)
        
        # Boucle ppo
        actions_tensor = torch.as_tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        
        # on normalise l'avantage pour la stabilisation
        #advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        for epoch in range(epochs):
            for start in range(0, len(states_tensor), batch_size):
                end = start + batch_size
                
                batch_states = states_tensor[start:end]
                batch_actions = actions_tensor[start:end]
                batch_old_log_probs = old_log_probs_tensor[start:end]
                batch_advantages = advantages_tensor[start:end]
                batch_returns = returns_tensor[start:end]
                
                # forward policy
                logits = self.policy_network(batch_states)
                if not torch.isfinite(logits).all():
                    # on évite de créer une distribution invalide
                    print("Nan/Inf détecté dans logits, batch ignoré")
                    continue
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # value loss
                values_pred = self.value_network(batch_states).squeeze(-1)
                value_loss = nn.functional.mse_loss(values_pred, batch_returns)
                
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
                self.policy_optimizer.step()
                self.value_optimizer.step()
        
        # clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


def run_training(env: AgentEnv, agent: AgentPPO, episodes: int, rollout_length: int, update_epochs: int, batch_size: int = 64) -> List[float]:
    """Boucle d'entraînement simple."""
    rewards_history: List[float] = []
    for episode_idx in range(episodes):

        task = random.choice(all_tasks)
        obs = env.reset(task)
        state = obs
        episode_reward = 0.0
        print(f"[episode {episode_idx + 1}/{episodes}] start task={task!r}")

        for step_idx in range(rollout_length):
            action_idx, value, log_prob = agent.select_action(state, training=True)
            next_obs, reward, done = env.step(action_idx)

            agent.store_transition(state, action_idx, reward, value, log_prob, done)

            episode_reward += reward
            state = next_obs
            print(
                f"  step={step_idx + 1}/{rollout_length} action={action_idx} "
                f"reward={reward:.4f} done={done} "
                f"dev={env.nb_dev_calls} analyst={env.nb_analyst_calls} reviewer={env.nb_reviewer_calls} "
                f"quality={env.current_quality_score:.3f}"
            )
            if done:
                break

        agent.update_policy(update_epochs, batch_size=batch_size)

        rewards_history.append(episode_reward)
        print(f"[episode {episode_idx + 1}/{episodes}] end total_reward={episode_reward:.4f}")

    return rewards_history
