"""Env pour le dag dynamique avec RL."""
import os
import subprocess
import json
from typing import Any, Dict, List
from llm_manager import LLMManager


class DAGEnv:
    def __init__(self, llm_manager: LLMManager, max_steps: int = 50, step_penalty: float = 0.01, workdir: str ="."):
        self.llm = llm_manager
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.graph: list = []
        self.steps = 0
        self.done = False
        self.workdir = workdir
        
        self.act_dim = 3 # CODER TESTER STOP
        
        self.obs_dim = 3 # 

    def reset(self) -> Dict[str, Any]:

        self.graph = []
        self.steps = 0
        self.done = False
        return self._get_observation()
    
    def _build_prompt(self, role: str) -> str:
        history = "\n".join(
            f"{r}: {t[:400]}" for r, t in self.graph
        )

        if role == "CODER":
            return f"""
    You are a coding assistant.
    Modify or add code to make all unit tests pass.

    Current history:
    {history}

    Return ONLY a JSON:
    {{"files": {{"path/to/file.py": "file content"}}}}
    """

        if role == "TESTER":
            return f"""
    You are a testing assistant.
    Write or improve unit tests to catch bugs.

    Current history:
    {history}

    Return ONLY a JSON:
    {{"files": {{"tests/test_x.py": "file content"}}}}
    """
    
    def apply_output(output: Dict[str, Any], workspace_dir: str) -> List[str]:
        
        written = []
        files = output.get("files", {})
        if not isinstance(files, dict):
            return written

        for rel_path, content in files.items():
            if not isinstance(rel_path, str) or not isinstance(content, str):
                continue
            # sécurité minimale
            if rel_path.startswith("/") or rel_path.startswith("\\"):
                continue
            if ".." in rel_path.replace("\\", "/").split("/"):
                continue

            full_path = os.path.join(workspace_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            written.append(rel_path)

        return written

    def step(self, action):

        if action == "STOP" :
            # On termine et on score l'état actuel
            self.done = True
            self.steps += 1
            succes, log = self.run_pytest()
            if succes :
                reward = 1.0
            else:
                reward = -1.0
            done = True
            info = {"pytest_passed": succes, "pytest_log": log}
            
            return self._get_observation(), reward, done, info

        # 1 on construit le prompt
        prompt = self._build_prompt(action)
        
        # 2 on appel le llm
        text = self.llm.generate_text(prompt=prompt, temperature=0.2, max_tokens=1200)
        
        # 3 on tiens un historique avec le graphe
        self.graph.append((action,text))
        
        # 4  On applique et on écrit sur le disque
        try:
            output = json.loads(text)
        except json.JSONDecodeError:
            output = {"files": {}}
        
        written = self.apply_output(output, self.workdir)
        
        # 5 On calcul la reward
        succes, log = self.run_pytest()
        if succes :
            reward = 1.0 - self.step_penalty
        else:
            reward = -1.0 - self.step_penalty
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
        
        return self._get_observation(), reward, self.done, {"pytest_passed": succes}
        
        
        
    def run_pytest(self):
        """Exécute rapidement la suite pytest et retourne (succès, sortie)."""
        try:
            result = subprocess.run(
                ["pytest", "-q"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            return False, f"pytest introuvable: {exc}"
        return result.returncode == 0, result.stdout + result.stderr

    def _get_observation(self) -> Dict[str, Any]:
        
        return {
            "graph": list(self.graph),
            "steps": self.steps,
            "done": self.done,
            "max_steps": self.max_steps,
        }
