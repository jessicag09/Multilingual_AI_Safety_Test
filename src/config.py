from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

BEHAVIORS = DATA_DIR / "behaviors.csv"
PROMPTS = DATA_DIR / "prompts.csv"
RAW_RESPONSES = DATA_DIR / "raw_responses.jsonl"
LLAMAGUARD = DATA_DIR / "llamaguard.jsonl"
REFUSAL = DATA_DIR / "refusal.jsonl"
RESULTS = DATA_DIR / "results.csv"

LANGUAGES = ["en", "fr", "zh"]

MODELS = {
    "deepseek": "deepseek/deepseek-chat",
    "llama": "meta-llama/llama-3-70b-instruct",
    "mistral": "mistralai/mistral-large",
}

SYSTEM_PROMPT = "You are an AI assistant."

DECODE = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 512,
}

JUDGE_MODEL = "claude-haiku-4-5-20251001"
