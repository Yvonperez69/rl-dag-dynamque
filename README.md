# rl-dag-dynamque
"""
Env pour le dag dynamique avec rl

"""

pour la reward 
- si le code compile
- llm juge 
- 


build prompt 
    def build_prompt(self, context: Dict[str, Any]) -> str:
    
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