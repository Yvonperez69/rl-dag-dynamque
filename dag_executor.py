import os
from typing import Dict, Any, List, Optional
import json
import requests
from dataclasses import dataclass
from test_executor import TestExecutor, CodeValidator

"""
Un executeur du dag + appels LLM + écriture de fichiers
dabord on vas prendre un DAG simple et l'éxécuter dans l'ordre, 
appeler un llm par noeud et écrire du code dans un dossier workspace
"""

initial_graph = "/Users/yvonperez/Desktop/TRDATA/rl-dag-proto/results/inference/inference_result_20260130_162152.json"

with open("/Users/yvonperez/Desktop/TRDATA/rl-dag-proto/results/inference/inference_result_20260130_162152.json", "r", encoding="utf-8") as f:
    dag_test = json.load(f)
    
@dataclass # Ce décorateur indique à Python que cette classe doit être traitée comme une dataclass. ça sert à éviter d'écrire des __init__, __repr__ ect
class ExecutionResult :
    code_compiles: bool
    tests_generated: bool
    tests_passed: bool
    nb_tests: int
    errors: List[str]
    trace: Dict

class OllamaLLMManager:
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


def topological_sort(dag):
    
    """Fonction pour determiner l'ordre d'éxécution des noeuds"""
    
    order = [] # liste final
    graph = dag[0]["optimal_dag"]
    edges = graph["edges"]
    
    # on construit une liste d'adjacence et un compteur du nombre de dépendances pour chaque noeuds
    adj = [[] for _ in range(len(graph["nodes"]))]
    nb_dep = [0]*len(graph["nodes"])
    
    for u, v in edges:
        adj[u].append(v)
        nb_dep[v] += 1
    
    # on construit une stack de noeuds indépendants 
    stack = []
    for k in range(len(adj)):
        if nb_dep[k] == 0 :
            stack.append(k)
    
    while len(stack) != 0 :
        u = stack[0]
        order.append(u)
        stack.remove(u)
        
        for v in adj[u] :
            nb_dep[v] -= 1
            if nb_dep[v] == 0 :
                stack.append(v)
    
    return order

# on doit creer le contexte pour proceder à l'éxécution d'un noeud



def _list_workspace_files(workspace_dir: str) -> List[str]:
    """Liste tous les fichiers présents dans le workspace (chemins relatifs)."""
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


def _build_parents_map(edges: List[List[int]]) -> Dict[int, List[int]]:
    """edges = [[u,v],...] -> parents[v] = [u,...]"""
    parents: Dict[int, List[int]] = {}
    for e in edges:
        if not (isinstance(e, list) and len(e) == 2):
            continue
        u, v = e
        parents.setdefault(v, []).append(u)
    return parents


def build_run_node_context(
    graph_input: List[Dict[str, Any]],
    *,
    node_id: int,
    node_outputs: Dict[int, Dict[str, Any]],
    workspace_dir: str,
    task_index: int = 0,
    max_parent_items: int = 6,
) -> Dict[str, Any]:
    """
    Construit le contexte minimal pour exécuter UN nœud du DAG.

    Paramètres
    - graph_input: ta liste de tâches, ex: [ {task_id, task_description, optimal_dag, ...} ]
    - node_id: id du nœud qu'on veut exécuter
    - node_outputs: dict {node_id: output_struct} pour les nœuds déjà exécutés
    - workspace_dir: dossier où le code est généré (pour snapshot fichiers)
    - task_index: si tu as plusieurs items dans la liste
    - max_parent_items: limite pour éviter un prompt énorme

    Retour
    - dict context: {task, task_id, node, parents_outputs, workspace_files, ...}
    """
    if not isinstance(graph_input, list) or len(graph_input) == 0:
        raise ValueError("graph_input doit être une liste non vide du type que tu as montré.")

    item = graph_input[task_index]
    task_id = item.get("task_id")
    task_description = item.get("task_description", "")

    optimal_dag = item.get("optimal_dag") or {}
    nodes = optimal_dag.get("nodes") or []
    edges = optimal_dag.get("edges") or []

    # map id -> node
    node_map: Dict[int, Dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict) and "id" in n:
            node_map[int(n["id"])] = n

    if node_id not in node_map:
        raise KeyError(f"node_id={node_id} introuvable dans optimal_dag['nodes'].")

    parents_map = _build_parents_map(edges)
    parent_ids = parents_map.get(node_id, [])

    # On compresse les outputs parents (summary + fichiers écrits) pour rester léger
    parents_outputs_compact: List[Dict[str, Any]] = []
    for pid in parent_ids[:max_parent_items]:
        out = node_outputs.get(pid, {})
        files = out.get("files", {})
        parents_outputs_compact.append({
            "node_id": pid,
            "summary": out.get("summary"),
            "written_files": list(files.keys()) if isinstance(files, dict) else None,
            # fallback debug si l’output n’était pas structuré
            "raw": out.get("raw") if "summary" not in out else None,
        })

    context: Dict[str, Any] = {
        "task_id": task_id,
        "task": task_description,
        "workspace_dir": workspace_dir,
        "workspace_files": _list_workspace_files(workspace_dir),
        "node": node_map[node_id],                 # le node complet (role/action/etc.)
        "node_id": node_id,
        "parent_ids": parent_ids,
        "parents_outputs": parents_outputs_compact,
        
        "dag_metadata": {
            "quality_score": item.get("quality_score"),
            "metrics_scores": item.get("metrics_scores"),
            "complexity_level": item.get("complexity_level"),
        }
    }
    return context



def build_prompt(context: Dict[str, Any]) -> str:
    
    node = context.get("node", {}) or {}
    role = node.get("role", "agent")
    action = node.get("action", "do the task")
    priority = node.get("priority", "medium")
    task = context.get("task", "")
    workspace_files = context.get("workspace_files", []) or []
    parents_outputs = context.get("parents_outputs", []) or []

    role_guidance = ""

    if role == "analyst":
        role_guidance = (
            "- Clarify requirements, assumptions, and success criteria.\n"
            "- Produce concrete artifacts such as specs, requirements, or data contracts.\n"
            "- Prefer markdown or lightweight text files (e.g., docs/requirements.md).\n"
        )

    elif role == "developer":
        role_guidance = (
            "- Implement core functionality in runnable code.\n"
            "- Prefer simple, minimal implementations that run locally.\n"
            "- Organize code cleanly under src/.\n"
        )

    elif role == "tester":
        role_guidance = (
            "- Focus on verification and validation.\n"
            "- Create tests under tests/ (unit or integration tests).\n"
            "- Ensure tests are minimal and runnable.\n"
        )

    else:
        
        role_guidance = (
            "- Your role may be unfamiliar.\n"
            "- You MUST still produce concrete, usable files.\n"
            "- Infer appropriate deliverables directly from the action.\n"
            "- Prefer minimal, runnable, and verifiable outputs.\n"
        )

    
    # Exemple concret : JSON strict + newlines échappées
    example_json = (
        '{"summary":"Created a minimal runnable module.","files":{"src/main.py":"'
        'def main():\\n'
        '    print(\\"hello\\")\\n'
        '\\n'
        'if __name__ == \\"__main__\\":\\n'
        '    main()\\n'
        '"}}'
    )

    prompt = f"""
SYSTEM:
You are a coding agent collaborating in a DAG execution. Follow instructions exactly.

GLOBAL TASK:
{task}

CURRENT NODE:
- node_id: {context.get("node_id")}
- role: {role}
- action: {action}
- priority: {priority}

AVAILABLE CONTEXT:
- workspace files (paths only): {workspace_files}
- outputs from dependencies (compact): {parents_outputs}

ROLE GUIDANCE:
{role_guidance if role_guidance else "- Follow your action precisely.\n"}

OUTPUT REQUIREMENTS (MANDATORY):
1) You MUST return ONLY valid JSON. No markdown. No explanations.
2) DO NOT wrap your answer in code fences like ```json or ``` .
3) In JSON, each file content MUST be a valid JSON string:
   - Use \\n for new lines
   - Escape quotes as \\\"
   - DO NOT use Python triple quotes (\"\"\") anywhere.
4) If you violate the JSON format, the system will fail and your output will be discarded.

JSON SCHEMA:
- "summary": string
- "files": object mapping relative file paths to full file contents (as JSON strings)

EXACT EXAMPLE (copy this format):
{example_json}

RULES:
- 'files' keys are RELATIVE paths (e.g., "src/app.py"), no absolute paths, no "..".
- Keep dependencies minimal.
- If unsure, produce the simplest working implementation.
""".strip()

    return prompt



def run_node(context: Dict[str, Any], llm_manager: Any, *, temperature: float = 0.2, max_tokens: int = 5*1200,) -> Dict[str, Any]:
    """
    Exécute un nœud : construit le prompt, appelle le LLM, parse et valide l'output JSON.

    Retourne un dict avec au minimum:
      - summary (str|None)
      - files (dict)
      - raw (str) si parsing échoue (utile debug)
    """
    prompt = build_prompt(context)
    
    is_test: bool = True
    node = context.get("node", {}) or {}
    role = node.get("role", "agent")
    
    
        

    # Appel LLM 
    
    try:
        raw = llm_manager.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)
    except TypeError:
        raw = llm_manager.generate_text(prompt)

    # 1) Parse JSON strict
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": None, "files": {}, "raw": raw}

    if not isinstance(out, dict):
        return {"summary": None, "files": {}, "raw": raw}

    # 2) Normalisation minimale
    summary = out.get("summary")
    files = out.get("files", {})

    if not isinstance(summary, str):
        summary = None

    if not isinstance(files, dict):
        files = {}

    # 3) Sécurisation des chemins (empêche écriture hors workspace)
    safe_files: Dict[str, str] = {}
    for path, content in files.items():
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        if path.startswith("/") or path.startswith("\\"):
            continue
        if ".." in path.replace("\\", "/").split("/"):
            continue
        if path.strip() == "":
            continue
        safe_files[path] = content

    return {"summary": summary, "files": safe_files, "raw": raw}

    
def apply_output(output: Dict[str, Any], workspace_dir: str) -> List[str]:
    """
    Écrit sur disque les fichiers retournés par run_node().
    output attendu: {"files": { "path": "content", ... }, ...}
    """
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


def execute_dag_test(dag_input: List[Dict[str, Any]], llm_manager: Any, workspace_dir: str = "workspace_run_001",) -> ExecutionResult:
    """
    Exécute le DAG contenu dans dag_input (format exact de ton exemple).
    """
    os.makedirs(workspace_dir, exist_ok=True)

    # 1) ordre topo
    order = topological_sort(dag_input)

    # 2) exécution
    node_outputs: Dict[int, Dict[str, Any]] = {}
    for node_id in order:
        
        #création du contexte
        
        context = build_run_node_context(
            graph_input=dag_input,
            node_id=node_id,
            node_outputs=node_outputs,
            workspace_dir=workspace_dir,
        )

        out = run_node(context, llm_manager) # génération du code
      
        written = apply_output(out, workspace_dir) # écriture du code

        # on garde une trace utile pour les nœuds suivants
        out["written_files"] = written
        node_outputs[node_id] = out

        print(f"[node {node_id}] summary={out.get('summary')!r} written={len(written)}")
        
        print("===================================================================" + "\n")
        print(out['raw'])

    return node_outputs


class DAGExecutor:
    """Exécute un DAG complet et orchestre validation + tests."""
    
    def __init__(self, llm_manager: Any, workspace_dir: str = "workspace_run_001"):
        self.llm_manager = llm_manager
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.test_executor = TestExecutor(self.workspace_dir)
        self.code_validator = CodeValidator(self.workspace_dir)
    
    def execute(self, dag: Any, context: Dict[str, Any]) -> ExecutionResult:
        # 1. exécution des noeuds
        dag_input = self._normalize_dag_input(dag, context)
        execution_trace: Dict[str, Any] = {"nodes": [], "tests": None, "validation": None}
        errors: List[str] = []
        
        order = topological_sort(dag_input)
        node_outputs: Dict[int, Dict[str, Any]] = {}
        for node_id in order:
            node_context = build_run_node_context(
                graph_input=dag_input,
                node_id=node_id,
                node_outputs=node_outputs,
                workspace_dir=self.workspace_dir,
            )
            out = run_node(node_context, self.llm_manager)
            written = apply_output(out, self.workspace_dir)
            out["written_files"] = written
            node_outputs[node_id] = out
            execution_trace["nodes"].append({
                "node_id": node_id,
                "summary": out.get("summary"),
                "written_files": written,
            })
        
        # 2. génération du code (déjà écrite) + validation
        validation = self.code_validator.validate_all_files()
        execution_trace["validation"] = validation
        code_ok = bool(validation.get("valid")) and validation.get("files_checked", 0) > 0
        if not validation.get("valid"):
            for err in validation.get("syntax_errors", []):
                errors.append(f"Syntax error: {err.get('file')}: {err.get('error')}")
            for err in validation.get("import_errors", []):
                errors.append(f"Import error: {err.get('file')}: {err.get('missing_imports')}")
        
        # 3. génération des tests (si manquants)
        tests_generated = self._ensure_tests_exist()
        
        # 4. exécution des tests
        all_tests_passed = False
        nb_tests = 0
        if tests_generated:
            test_results = self.test_executor.run_all_tests()
            execution_trace["tests"] = test_results
            nb_tests = test_results.get("total_tests", 0)
            all_tests_passed = bool(test_results.get("success"))
            if not all_tests_passed:
                errors.append(test_results.get("message", "Tests failed"))
        else:
            errors.append("No tests were generated")
        
        return ExecutionResult(
            code_compiles=code_ok,
            tests_generated=tests_generated,
            tests_passed=all_tests_passed,
            nb_tests=nb_tests,
            errors=errors,
            trace=execution_trace
        )
    
    def _normalize_dag_input(self, dag: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(dag, list):
            return dag
        if isinstance(dag, dict):
            task_description = (
                context.get("task_description")
                or context.get("task")
                or dag.get("task_description")
                or ""
            )
            return [{
                "task_id": dag.get("task_id", "task_0"),
                "task_description": task_description,
                "optimal_dag": dag.get("optimal_dag", dag)
            }]
        raise ValueError("dag doit être une liste ou un dict compatible.")
    
    def _ensure_tests_exist(self) -> bool:
        existing = self.test_executor.find_test_files()
        if existing:
            return True
        return self._generate_tests_via_llm()
    
    def _generate_tests_via_llm(self) -> bool:
        code_summary = self._summarize_code_files()
        prompt = f"""
TASK: Generate pytest tests for the code below.

CODE SUMMARY:
{code_summary}

REQUIREMENTS:
1. Return ONLY valid JSON
2. JSON format: {{"summary": "...", "files": {{"tests/test_main.py": "..."}}}}
3. Tests must be runnable with pytest
"""
        try:
            raw = self.llm_manager.generate_text(prompt, temperature=0.2)
        except TypeError:
            raw = self.llm_manager.generate_text(prompt)
        
        output = self._parse_llm_json(raw)
        if output is not None:
            files = output.get("files", {})
            return self._write_files(files)
        
        # fallback: extraire bloc de code
        code_blocks = self._extract_code_blocks(raw)
        if code_blocks:
            return self._write_files({"tests/test_main.py": code_blocks[0]})
        
        # fallback minimal
        return self._write_files({"tests/test_smoke.py": "def test_smoke():\n    assert True\n"})
    
    def _write_files(self, files: Dict[str, str]) -> bool:
        wrote_any = False
        for rel_path, content in files.items():
            if not isinstance(rel_path, str) or not isinstance(content, str):
                continue
            if rel_path.startswith(("/", "\\")):
                continue
            if ".." in rel_path.replace("\\", "/").split("/"):
                continue
            full_path = os.path.join(self.workspace_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            wrote_any = True
        return wrote_any
    
    def _summarize_code_files(self, max_files: int = 5, max_chars: int = 800) -> str:
        parts: List[str] = []
        for root, _, files in os.walk(self.workspace_dir):
            for name in files:
                if not name.endswith(".py"):
                    continue
                if "test" in name:
                    continue
                full_path = os.path.join(root, name)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()[:max_chars]
                    rel_path = os.path.relpath(full_path, self.workspace_dir)
                    parts.append(f"File: {rel_path}\n{content}\n")
                except Exception:
                    continue
                if len(parts) >= max_files:
                    break
            if len(parts) >= max_files:
                break
        return "\n".join(parts) if parts else "No code files found"
    
    def _parse_llm_json(self, raw_output: str) -> Optional[Dict[str, Any]]:
        if not raw_output:
            return None
        cleaned = raw_output.strip()
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                if "{" in part and "}" in part:
                    cleaned = part.strip()
                    break
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = cleaned[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    
    def _extract_code_blocks(self, raw_output: str) -> List[str]:
        if not raw_output:
            return []
        blocks: List[str] = []
        parts = raw_output.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            if not block:
                continue
            if "\n" in block:
                first_line, rest = block.split("\n", 1)
                if first_line.strip().isalpha():
                    block = rest
            blocks.append(block.strip())
        return blocks


if __name__ == "__main__":
    # ⚠️ Mets le modèle que tu as réellement dans "ollama list"
    llm = OllamaLLMManager(
        base_url="http://localhost:11434",
        model="llama3.2:latest",
    )

    outputs = execute_dag_test(dag_test, llm, workspace_dir="workspace_dag_test")
    print("\nDONE. Workspace:", os.path.abspath("workspace_dag_test"))

    
