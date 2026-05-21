import httpx
from .base import ModelClient


class OllamaClient(ModelClient):
    URL = "http://localhost:11434/api/chat"

    NAME_MAP = {
        "deepseek/deepseek-chat": "deepseek-v2",
        "meta-llama/llama-3-70b-instruct": "llama3:70b",
        "mistralai/mistral-large": "mistral-large",
    }

    def generate(self, model_id, system, user, temperature, top_p, max_new_tokens):
        payload = {
            "model": self.NAME_MAP.get(model_id, model_id),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_new_tokens,
            },
            "stream": False,
        }
        r = httpx.post(self.URL, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]
