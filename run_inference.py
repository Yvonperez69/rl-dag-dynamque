from agent_env import AgentEnv
from llm_manager import LLMManager
from train_ppo import AgentPPO
import torch

def main():
    
    llm = LLMManager()
    
    inference_dir = "inference"
    # Environnement
    env = AgentEnv(llm, max_steps=30, workdir=inference_dir)
    
    # Agent
    agent = AgentPPO(env=env,
        lr=3e-4,
        gae_lambda=0.95,
        eps_clip=0.2,
        entropy_coef=0.01,
        gamma=0.99)
    
    # On charge la politique
    agent.policy_network.load_state_dict(torch.load("policy.pt"))
    agent.policy_network.eval()

    task = input("Enter coding tasks : ")
    obs = env.reset(task)
    
    done = False
    while not done:
        with torch.no_grad():
            action, _, _ = agent.select_action(obs, training=False)

        obs, reward, done = env.step(action)

if __name__ == "__main__":
    main()
