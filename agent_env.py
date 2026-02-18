"""Env pour le dag dynamique avec RL."""
import os
import py_compile
import json
import numpy as np
from typing import Any, Dict, List
import sys
from pathlib import Path

from sympy import beta
from llm_manager import LLMManager
import re
import subprocess


class AgentEnv:
    def __init__(self, llm_manager: LLMManager, max_steps: int = 50, workdir: str ="training"):

        self.llm = llm_manager
        self.max_steps = max_steps

        #context minimal pour les agents
        self.task_description = None
        self.workdir = workdir
        self.hist = []

        # trajectoire RL
        self.actions = []
        self.rewards = []
        self.current_quality_score = 0.0
        self.nb_test = 0 # 1 si le test réussi et 0 sinon
        self.steps = 0
        self.done = False
        self.nb_tester_calls = 0
        self.nb_dev_calls = 0
        self.nb_analyst_calls = 0
        self.nb_reviewer_calls = 0

        self.act_dim =  5 # appeler Analyst, appeler Dev, appeler Reviewer, appeler tester, terminer l'épisode
        self.obs_dim = 5 # state, action, reward, nombre de steps, done

    def reset(self, task_description):
        self.task_description = task_description
        self.hist = []
        self.actions = []
        self.rewards = []
        self.nb_dev_calls = 0
        self.nb_analyst_calls = 0
        self.nb_reviewer_calls = 0
        self.nb_tester_calls = 0
        self.steps = 0
        self.current_quality_score = 0.0
        self.done = False
        return self._get_observation()
    
    def _list_workspace_files(self,workspace_dir: str) -> List[str]:
    
        if not os.path.exists(workspace_dir):
            return []
        out: List[str] = []
        for root, _, files in os.walk(workspace_dir):
            for name in files:
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, workspace_dir)
                out.append(rel_path)
        out.sort()
        return out

    def build_context(self, hist, workspace_dir):
        """
        Construit le contexte minimal pour faire appel à un agent.

        Paramètres
        - historique 
        - workspace_dir: dossier où le code est généré (pour snapshot fichiers)

        Retour
        - dict context: {task, hist, workspace_dir, workspace_files, ...}
        """
 
        context = {
            "task": self.task_description,
            "hist": hist[-3:] if len(hist) >= 3 else hist,
            "workspace_dir": workspace_dir,
            "workspace_files": self._list_workspace_files(workspace_dir),

        }
        return context
    
    def apply_output(self, output: Any, workspace_dir: str) -> List[str]:
        """
        Écrit sur disque les sorties LLM.
        Supporte:
        - dict au format {"files": {"path": "content", ...}}
        - str (code brut) -> écrit dans un fichier cible heuristique
        """
        written: List[str] = []

        # 1) Si c'est un dict avec "files", comportement existant
        if isinstance(output, dict):
            files = output.get("files", {})
            if isinstance(files, dict):
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

            # Si le dict ressemble à du JSON brut dans un champ "raw"
            raw = output.get("raw")
            if isinstance(raw, str):
                output = raw

        # 2) Si c'est une string (sortie brute des call_*)
        if isinstance(output, str):
            content = output.strip()
            if content == "":
                return written

            # Tentative JSON (compatibilité avec d'autres appels LLM)
            if content.startswith("{") and "\"files\"" in content:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and isinstance(parsed.get("files"), dict):
                    return self.apply_output(parsed, workspace_dir)

            # Heuristique de fichier cible: si un seul .py existe, on l'écrase
            workspace_files = self._list_workspace_files(workspace_dir)
            py_files = [p for p in workspace_files if p.endswith(".py")]

            if len(py_files) == 1:
                target_rel = py_files[0]
            elif "main.py" in workspace_files:
                target_rel = "main.py"
            elif "src/main.py" in workspace_files:
                target_rel = "src/main.py"
            else:
                # fallback sûr pour éviter d'écraser un fichier existant
                base = "llm_output.py"
                target_rel = base
                if target_rel in workspace_files:
                    i = 1
                    while f"llm_output_{i}.py" in workspace_files:
                        i += 1
                    target_rel = f"llm_output_{i}.py"

            full_path = os.path.join(workspace_dir, target_rel)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            written.append(target_rel)

        return written

    def extract_score_from_llm(self, output: str) -> float:
        try:
            # 1️⃣ Extraire le bloc JSON entre accolades
            match = re.search(r"\{.*?\}", output, re.DOTALL)
            if not match:
                return 0.0
            
            json_str = match.group()
            
            # 2️⃣ Parser le JSON
            parsed = json.loads(json_str)
            
            return float(parsed.get("score", 0.0))
        
        except Exception:
            return 0.0

    def check_syntax(self, folder):
        try:
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.endswith(".py"):
                        path = os.path.join(root, f)
                        py_compile.compile(path, doraise=True)
            return True
        except py_compile.PyCompileError:
            return False

        
    def run_tests(self, workdir):
        workdir_path = Path(workdir)
        if not workdir_path.exists():
            return False

        # 1) Run every discovered pytest file (Windows-safe: python -m pytest)
        test_files = sorted(
            [str(p.relative_to(workdir_path)) for p in workdir_path.rglob("test_*.py")]
            + [str(p.relative_to(workdir_path)) for p in workdir_path.rglob("*_test.py")]
        )
        # Deduplicate while preserving order
        test_files = list(dict.fromkeys(test_files))

        if test_files:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *test_files],
                cwd=workdir,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0

        # 2) Fallback: run a likely entry script if no tests are present
        candidates = ["main.py", "solution.py", "app.py"]
        for name in candidates:
            path = workdir_path / name
            if path.exists():
                result = subprocess.run(
                    [sys.executable, str(path)],
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                )
                return result.returncode == 0

        # 3) Last fallback: if exactly one Python file exists, run it
        py_files = sorted(workdir_path.rglob("*.py"))
        if len(py_files) == 1:
            result = subprocess.run(
                [sys.executable, str(py_files[0])],
                cwd=workdir,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0

        return False



    def compute_reward(self, context):
        reward = 0.1 # reward de base 

        workspace_dir = context.get("workspace_dir")

        # Reward basé sur l'EXECUTION : nb_test passé, temps d'éxécution, compilation
        reward_execution = 0.0
        if self.check_syntax(workspace_dir) :
            reward_execution += 1.0
            if self.nb_tester_calls > 0 :
                self.nb_test = 1 if self.run_tests(workspace_dir) else 0
                reward_execution += self.nb_test

        # temps d'exec

        reward += reward_execution

        reward -= 0.01 # step penalty

        # appel au llm juge
        prev_q = self.current_quality_score
        reward_llm = self.extract_score_from_llm(self.llm.call_judge(context))
        reward += (reward_llm - prev_q)/10 #On normalise la reward
        self.current_quality_score  = reward_llm
        return reward

    def step(self, action):
        
        reward = 0.0

        if action == 4:
            # On termine et on score l'état actuel

            self.steps += 1
            self.done = True

            return self._get_observation(), reward, self.done

        if action == 0: # ANALYSTE

            context = self.build_context(self.hist, self.workdir)
            out = self.llm.call_analyst(context)
            #on tiens un historique pour donner la sortie de l'analyste au développeur/reviewer
            self.hist.append({"action": action, "output": out})
            self.nb_analyst_calls += 1
            
        if action == 1: # DEVELOPER

            context = self.build_context(self.hist, self.workdir)
            out = self.llm.call_developer(context)
            #on tiens un historique 
            self.hist.append({"action": action, "output": out})

            written = self.apply_output(out, self.workdir)

            self.nb_dev_calls += 1

        if action == 2: #REVIEWER

            context = self.build_context(self.hist, self.workdir)
            out = self.llm.call_reviewer(context)
            #on tiens un historique
            self.hist.append({"action": action, "output": out})

            written =  self.apply_output(out, self.workdir)

            self.nb_reviewer_calls += 1
        
        if action == 3: # TESTER

            context = self.build_context(self.hist, self.workdir)
            out = self.llm.call_tester(context)
            #on tiens un historique
            self.hist.append({"action": action, "output": out})

            written =  self.apply_output(out, self.workdir)

            self.nb_tester_calls += 1

        #On calcul la reward

        self.steps += 1
        reward = self.compute_reward(context)
        
        return self._get_observation(), reward, self.done
        

    def _get_observation(self):
        return np.array([
            self.steps / self.max_steps,
            self.nb_dev_calls / 10,
            self.nb_analyst_calls / 10,
            self.nb_reviewer_calls / 10,
            self.current_quality_score
        ], dtype=np.float32)
