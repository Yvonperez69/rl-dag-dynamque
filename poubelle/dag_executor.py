import os
from typing import Dict, Any, List, Optional
import json
import requests
from llm_manager import LLMManager
from dataclasses import dataclass


"""
Un executeur du dag + appels LLM + écriture de fichiers
dabord on vas prendre un DAG simple et l'éxécuter dans l'ordre, 
appeler un llm par noeud et écrire du code dans un dossier workspace
"""


initial_graph = [{
  "dag": {
    "nodes": [
      {
        "id": 0,
        "role": "analyst"
      },
      {
        "id": 1,
        "role": "developer"
      },
      {
        "id": 2,
        "role": "tester"
      }
    ],
    "edges": [
      [0, 1],
      [1, 2]
    ]
  }
}]


def topological_sort(dag):
    
    """Fonction pour determiner l'ordre d'éxécution des noeuds"""
    
    order = [] # liste final
    graph = dag[0]["dag"]
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
    max_parent_items: int = 6,
) -> Dict[str, Any]:
    """
    Construit le contexte minimal pour exécuter UN nœud du DAG.

    Paramètres
    - graph_input: ta liste de tâches, ex: [ {task_id, task_description, optimal_dag, ...} ]
    - node_id: id du nœud qu'on veut exécuter
    - node_outputs: dict {node_id: output_struct} pour les nœuds déjà exécutés
    - workspace_dir: dossier où le code est généré (pour snapshot fichiers)
    - max_parent_items: limite pour éviter un prompt énorme

    Retour
    - dict context: {node, parents_outputs, workspace_files, ...}
    """
    if not isinstance(graph_input, list) or len(graph_input) == 0:
        raise ValueError("graph_input doit être une liste non vide du type que tu as montré.")

    task_block = graph_input[0]
    dag = task_block.get("dag", {})
    nodes = dag.get("nodes", []) 
    edges = dag.get("edges", [])
                                                                       
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
        "task": dag.get("task_description"),
        "workspace_dir": workspace_dir,
        "workspace_files": _list_workspace_files(workspace_dir),
        "node": node_map[node_id],                 # le node complet (role/action/etc.)
        "node_id": node_id,
        "parent_ids": parent_ids,
        "parents_outputs": parents_outputs_compact
    }
    return context



def build_prompt(context: Dict[str, Any]) -> str:
    
    node = context.get("node", {}) or {}
    role = node.get("role")
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



def run_node(context: Dict[str, Any], llm_manager: Any, *, temperature: float = 0.2, max_tokens: int = 1200,) -> Dict[str, Any]:
    """
    Exécute un nœud : construit le prompt, appelle le LLM, parse et valide l'output JSON.

    Retourne un dict avec au minimum:
      - summary (str|None)
      - files (dict)
      - raw (str) si parsing échoue (utile debug)
    """

    # Construire le prompt
    prompt = build_prompt(context)
    
    
    node = context.get("node", {}) or {}
    
    # Appel LLM 
    
    try:
        raw = llm_manager.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)
    except TypeError:
        raw = llm_manager.generate_text(prompt)

    # Parse JSON strict
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": None, "files": {}, "raw": raw}

    if not isinstance(out, dict):
        return {"summary": None, "files": {}, "raw": raw}

    #  Normalisation minimale
    summary = out.get("summary")
    files = out.get("files", {})

    if not isinstance(summary, str):
        summary = None

    if not isinstance(files, dict):
        files = {}

    # Sécurisation des chemins (empêche écriture hors workspace)
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


def execute_dag_test(dag_input: List[Dict[str, Any]], llm_manager: Any, workspace_dir: str = "workspace_run_001",) -> Dict[int, Dict[str, Any]]:
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
        

    return node_outputs


if __name__ == "__main__":
    
    user_prompt = input("Enter your prompt : ").strip()
    
    initial_graph[0]["task_description"] = user_prompt

    print(initial_graph)
    llm = LLMManager(
        base_url="http://localhost:11434",
        model="llama3.2:latest",
    )



    outputs = execute_dag_test(initial_graph, llm, workspace_dir="workspace_dag_test")
    #print("\nDONE. Workspace:", os.path.abspath("workspace_dag_test"))

    
