import argparse
import json
import os
import subprocess
from typing import Dict, List, Tuple

import torch

from agent_env import DAGEnv
from llm_manager import LLMManager
from train_ppo import AgentPPO, encode_obs


def load_last_run(save_dir: str) -> Dict[str, object]:
    path = os.path.join(save_dir, "last_run.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config introuvable: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_dot(graph: List[Tuple[str, str]], max_chars: int = 120) -> str:
    lines = ["digraph G {", "  rankdir=LR;"]
    prev = "start"
    lines.append('  start [label="START"];')
    for i, (role, text) in enumerate(graph, 1):
        snippet = text.replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3] + "..."
        node = f"n{i}"
        label = f"{role}: {snippet}".replace('"', "'")
        lines.append(f'  {node} [label="{label}"];')
        lines.append(f"  {prev} -> {node};")
        prev = node
    lines.append('  end [label="END"];')
    lines.append(f"  {prev} -> end;")
    lines.append("}")
    return "\n".join(lines)


def save_graph(output_dir: str, episode: int, graph: List[Tuple[str, str]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dot = render_dot(graph)
    dot_path = os.path.join(output_dir, f"graph_ep{episode}.dot")
    png_path = os.path.join(output_dir, f"graph_ep{episode}.png")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot)
    try:
        subprocess.run(
            ["dot", "-Tpng", dot_path, "-o", png_path],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Si Graphviz n'est pas disponible, on laisse le .dot.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualiser le graphe produit par PPO")
    parser.add_argument("--save-dir", type=str, default="runs", help="Dossier de sauvegarde")
    parser.add_argument("--output-dir", type=str, default="graphs", help="Dossier de sortie des graphes")
    parser.add_argument("--episodes", type=int, default=3, help="Nombre d'episodes")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="infer", help="Mode execution")
    parser.add_argument("--rollout-length", type=int, default=20, help="Steps max par episode (train)")
    parser.add_argument("--update-epochs", type=int, default=4, help="Passes PPO par update (train)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch PPO (train)")
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
    if os.path.exists(policy_path):
        agent.policy_network.load_state_dict(torch.load(policy_path, map_location=agent.device))
    if os.path.exists(value_path):
        agent.value_network.load_state_dict(torch.load(value_path, map_location=agent.device))
    agent.policy_network.eval()
    agent.value_network.eval()

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        state = encode_obs(obs)
        done = False
        steps = 0

        while not done:
            action_idx, value, log_prob = agent.select_action(state, training=(args.mode == "train"))
            action_label = agent.action_labels[action_idx]
            next_obs, reward, done, _ = env.step(action_label)

            if args.mode == "train":
                agent.store_transition(state, action_idx, reward, value, log_prob, done)
                steps += 1
                if steps >= args.rollout_length or done:
                    agent.update_policy(args.update_epochs, batch_size=args.batch_size)
                    steps = 0

            state = encode_obs(next_obs)

        save_graph(args.output_dir, ep, env.graph)
        print(f"Episode {ep}: graph saved")


if __name__ == "__main__":
    main()
