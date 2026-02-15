import os
import torch
from agent_env import AgentEnv
from llm_manager import LLMManager
from train_ppo import AgentPPO
from train_ppo import run_training

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    llm = LLMManager()
    
    training_dir = "training"
    os.makedirs(training_dir, exist_ok=True)
    env = AgentEnv(llm, max_steps=30, workdir=training_dir)
    
    agent = AgentPPO(
        env=env,
        lr=3e-4,
        gae_lambda=0.95,
        eps_clip=0.2,
        entropy_coef=0.01,
        gamma=0.99
    )

    rewards = run_training(
        env=env,
        agent=agent,
        episodes=200,
        rollout_length=20,
        update_epochs=4,
        batch_size=32
    )

    torch.save(agent.policy.state_dict(), "policy.pt")

    print("Training finished.")
    print("Final reward:", rewards[-1])


if __name__ == "__main__":
    main()
