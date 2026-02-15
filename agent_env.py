"""Env pour le dag dynamique avec RL."""
import os
import subprocess
import json
import numpy as np
from typing import Any, Dict, List
from llm_manager import LLMManager


class AgentEnv:
    def __init__(self, llm_manager: LLMManager, task_description: str, max_steps: int = 50, workdir: str ="."):

        self.llm = llm_manager
        self.max_steps = max_steps

        #context minimal pour les agents
        self.task_description = task_description
        self.workdir = workdir
        self.hist = []

        # trajectoire RL
        self.actions = []
        self.rewards = []
        self.current_quality_score = 0
        self.steps = 0
        self.done = False
        self.nb_dev_calls = 0
        self.nb_analyst_calls = 0
        self.nb_reviewer_calls = 0

        self.act_dim =  4 # appeler Analyst, appeler Dev, appeler Reviewer, terminer l'épisode
        self.obs_dim = 5 # state, action, reward, nombre de steps, done

    def reset(self) -> Dict[str, Any]:

        self.actions = []
        self.rewards = []
        self.steps = 0
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
        - max_parent_items: limite pour éviter un prompt énorme

        Retour
        - dict context: {node, parents_outputs, workspace_files, ...}
        """

        context = {
            "task": self.task_description,
            "hist": hist[-3:] if len(hist) >= 3 else hist,
            "workspace_dir": workspace_dir,
            "workspace_files": self._list_workspace_files(workspace_dir),

        }
        return context
    
    def apply_output(self, output: Dict[str, Any], workspace_dir: str) -> List[str]:
        
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

        if action == 3:
            # On termine et on score l'état actuel
            done = True
            self.steps += 1
            
            
            return self._get_observation()

        if action == 0: # ANALYSTE

            context = self.build_context(self.hist, self.workdir)
            out = self.llm.call_analyst(context, self.task_description)
            #on tiens un historique pour donner la sortie de l'analyste au développeur/reviewer
            self.hist.append({"action": action, "output": out})
            self.nb_analyst_calls += 1
            
        if action == 1: # DEVELOPER

            context = self.build_context(self.hist, self.workdir)
            out = self.llm.call_developer(context, self.task_description)
            #on tiens un historique 
            self.hist.append({"action": action, "output": out})

            written = self.apply_output(out, self.workdir)

            self.nb_dev_calls += 1

        if action == 2: #REVIEWER

            context = self.build_context(self.hist, self.workdir)
            out = self.llm.call_reviewer(context, self.task_description)
            #on tiens un historique
            self.hist.append({"action": action, "output": out})

            written =  self.apply_output(out, self.workdir)

            self.nb_reviewer_calls += 1

        #On calcul la reward
        reward = 0.1 # reward de base

        self.steps += 1 
        reward -= 0.01 # step penalty

        # appel au llm juge
        prev_q = self.current_quality_score

        reward_llm = float(self.llm.call_judge(context).get("score"))
        reward += (reward_llm - prev_q)/10 #On normalise la reward
        self.current_quality_score  = reward_llm

        return self._get_observation(), reward, self.done
        

    def _get_observation(self):
        return np.array([
            self.steps / self.max_steps,
            self.nb_dev_calls / 10,
            self.nb_analyst_calls / 10,
            self.nb_reviewer_calls / 10,
            self.current_quality_score
        ], dtype=np.float32)
