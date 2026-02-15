from agent_env import AgentEnv
from llm_manager import LLMManager
from train_ppo import AgentPPO
from dataset import all_tasks
import random


task = random.choice(all_tasks)
