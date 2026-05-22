import os
import httpx
from .base import ModelClient


class TogetherClient(ModelClient):
    URL = "https://api.together.xyz/v1/chat/completions"

    def __init__(self):
        key = os.environ.get("TOGETHER_API_KEY")
        if not key:
            raise RuntimeError("TOGETHER_API_KEY not set")
        self.headers = {"Authorization": f"Bearer {key}"}

    def generate(self, model_id, system, user, temperature, top_p, max_new_tokens):
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        r = httpx.post(self.URL, json=payload, headers=self.headers, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
