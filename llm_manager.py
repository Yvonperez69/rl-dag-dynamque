import requests
from typing import Dict, Any, List
import os  

class LLMManager:
    """
    LLM manager minimal pour Ollama.
    Nécessite:
      - ollama serve
      - le modèle pull (ex: ollama pull llama3.2:latest)
    """
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:latest"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate_text(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1200) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
            # Ollama n'utilise pas toujours max_tokens selon versions/modèles, mais ça ne gêne pas.
            "max_tokens": max_tokens,
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
    
    def call_judge(self, context):

        hist = context.get("hist", [])
        task = context.get("task", "")
        workspace_files = context.get("workspace_files", [])
        parents_outputs = context.get("parents_outputs", [])

        prompt = f"""
        Evaluate the final solution.

        AVAILABLE CONTEXT in {context}:
        - history : {hist}
        - global task description : {task}
        - workspace files (paths only): {workspace_files}
        - outputs from dependencies (compact): {parents_outputs}

        Criteria:
        1. Correctness (0-4)
        2. Completeness (0-3)
        3. Clarity (0-2)
        4. Logical coherence (0-1)

        Return ONLY a JSON:
        {
        "score": float
        }

        """
        return prompt


    def call_analyst(self, context):

        hist = context.get("hist", [])
        task = context.get("task", "")
        workspace_files = context.get("workspace_files", [])
        parents_outputs = context.get("parents_outputs", [])

        prompt = f"""
    SYSTEM:
    You are a coding agent collaborating and You are a senior software architect and static code analyst. Follow instructions exactly.

    GLOBAL TASK:
    {task}
    

    Your missions are strictly:

    1) Understand the task precisely.
    2) Prepare a clear and structured prompt for the Developer agent.
    3) Verify whether the current code compiles and has valid syntax (if code exists).
    4) Evaluate whether the existing solution is optimal in terms of time and space complexity.
    5) Add meaningful comments to improve readability and maintainability.

    Instructions:

    - If no code exists, focus on understanding the problem and preparing the best possible structured instructions for the Developer.
    - If code exists:
        - Check syntax validity.
        - Detect logical issues.
        - Analyze time and space complexity.
        - Suggest improvements if a better algorithm exists.
        - Return a commented version of the code.




    AVAILABLE CONTEXT in {context}:
    - history : {hist}
    - global task description : {task}
    - workspace files (paths only): {workspace_files}
    - outputs from dependencies (compact): {parents_outputs}

    """
        return prompt

    def call_developer(self, context):

        hist = context.get("hist", [])
        task = context.get("task", "")
        workspace_files = context.get("workspace_files", [])
        parents_outputs = context.get("parents_outputs", [])

        prompt = f"""
        You are an expert Python developer.

        Your goal is to produce correct, clean, and executable Python code.

        Requirements:
        - The code must be fully runnable.
        - No placeholders.
        - No explanations.
        - No markdown.
        - Output only raw Python code.
        - Handle edge cases.
        - Follow the provided plan if available.

        AVAILABLE CONTEXT in {context}:
        - history : {hist}
        - global task description : {task}
        - workspace files (paths only): {workspace_files}
        - outputs from dependencies (compact): {parents_outputs}
        Return ONLY valid Python code.

        """

        return prompt

    def call_reviewer(self, context):

        hist = context.get("hist", [])
        task = context.get("task", "")
        workspace_files = context.get("workspace_files", [])
        parents_outputs = context.get("parents_outputs", [])

        prompt = f"""
        You are a strict senior code reviewer.

        Your task:
        - Detect logical errors.
        - Detect missing edge cases.
        - Improve robustness.
        - Fix potential runtime errors.
        - Improve clarity if needed.

        If the code is correct, return the improved full version.
        If there are issues, return a corrected full version.

        AVAILABLE CONTEXT in {context}:
        - history : {hist}
        - global task description : {task}
        - workspace files (paths only): {workspace_files}
        - outputs from dependencies (compact): {parents_outputs}

        Do NOT explain.
        Return only raw Python code.
        """

        return prompt