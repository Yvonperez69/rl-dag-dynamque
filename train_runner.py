import argparse
import random
from typing import Optional

import numpy as np
import torch

from dag_env import DAGEnv
from llm_manager import LLMManager
from train_ppo import AgentPPO, run_training


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on DAGEnv")
    parser.add_argument("--episodes", type=int, default=10, help="Nombre d'episodes")
    parser.add_argument("--rollout-length", type=int, default=20, help="Steps par episode avant update")
    parser.add_argument("--update-epochs", type=int, default=4, help="Passes PPO par update")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch PPO")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Coef d'entropie")
    parser.add_argument("--max-steps", type=int, default=50, help="Pas max par episode")
    parser.add_argument("--step-penalty", type=float, default=0.01, help="Penalite par pas")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL du serveur Ollama")
    parser.add_argument("--ollama-model", type=str, default="llama3.2:latest", help="Modele Ollama")
    parser.add_argument("--workdir", type=str, default=".", help="Repertoire de travail pour les fichiers generes")
    args = parser.parse_args()

    set_seed(args.seed)

    llm = LLMManager(base_url=args.ollama_url, model=args.ollama_model)
    env = DAGEnv(llm_manager=llm, max_steps=args.max_steps, step_penalty=args.step_penalty, workdir=args.workdir)
    agent = AgentPPO(
        env=env,
        lr=args.lr,
        gae_lambda=args.gae_lambda,
        eps_clip=args.eps_clip,
        entropy_coef=args.entropy_coef,
        gamma=args.gamma,
    )

    rewards = run_training(
        env=env,
        agent=agent,
        episodes=args.episodes,
        rollout_length=args.rollout_length,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
    )

    for i, r in enumerate(rewards, 1):
        print(f"Episode {i}: reward={r:.3f}")


if __name__ == "__main__":
    main()
