import requests

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