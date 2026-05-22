from abc import ABC, abstractmethod


class ModelClient(ABC):
    @abstractmethod
    def generate(self, model_id: str, system: str, user: str,
                 temperature: float, top_p: float, max_new_tokens: int) -> str:
        ...


def get_client(backend: str) -> ModelClient:
    if backend == "together":
        from .together import TogetherClient
        return TogetherClient()
    if backend == "openrouter":
        from .openrouter import OpenRouterClient
        return OpenRouterClient()
    if backend == "ollama":
        from .ollama import OllamaClient
        return OllamaClient()
    raise ValueError(f"unknown backend: {backend}")
