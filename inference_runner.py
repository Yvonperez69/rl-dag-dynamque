import argparse
import json
import os
from typing import Dict

import torch

from dag_env import DAGEnv
from llm_manager import LLMManager
from train_ppo import AgentPPO, encode_obs


def load_last_run(save_dir: str) -> Dict[str, object]:
    path = os.path.join(save_dir, "last_run.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config introuvable: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference PPO a partir du dernier run")
    parser.add_argument("--save-dir", type=str, default="runs", help="Dossier de sauvegarde")
    parser.add_argument("--episodes", type=int, default=3, help="Nombre d'episodes d'inference")
    args = parser.parse_args()

    cfg = load_last_run(args.save_dir)

    llm = LLMManager(base_url=str(cfg["ollama_url"]), model=str(cfg["ollama_model"]))
    env = DAGEnv(
        llm_manager=llm,
        max_steps=int(cfg["max_steps"]),
        step_penalty=float(cfg["step_penalty"]),
        workdir=str(cfg["workdir"]),
    )
    agent = AgentPPO(
        env=env,
        lr=float(cfg["lr"]),
        gae_lambda=float(cfg["gae_lambda"]),
        eps_clip=float(cfg["eps_clip"]),
        entropy_coef=float(cfg["entropy_coef"]),
        gamma=float(cfg["gamma"]),
    )

    policy_path = os.path.join(args.save_dir, "ppo_policy.pt")
    value_path = os.path.join(args.save_dir, "ppo_value.pt")
    agent.policy_network.load_state_dict(torch.load(policy_path, map_location=agent.device))
    agent.value_network.load_state_dict(torch.load(value_path, map_location=agent.device))
    agent.policy_network.eval()
    agent.value_network.eval()

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        state = encode_obs(obs)
        total_reward = 0.0
        done = False

        while not done:
            action_idx, value, log_prob = agent.select_action(state, training=False)
            action_label = agent.action_labels[action_idx]
            next_obs, reward, done, info = env.step(action_label)
            total_reward += reward
            state = encode_obs(next_obs)

        print(f"Episode {ep}: reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
