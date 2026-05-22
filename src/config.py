from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"

PROMPTS = DATA_DIR / "prompts.csv"
RAW_RESPONSES = DATA_DIR / "raw_responses.jsonl"
LLAMAGUARD = DATA_DIR / "llamaguard.jsonl"
REFUSAL = DATA_DIR / "refusal.jsonl"
RESULTS = DATA_DIR / "results.csv"

CULTURAL_PROBE_SOURCE = PROMPTS_DIR / "cultural_probe.json"
CULTURAL_PROBE_PROMPTS = DATA_DIR / "cultural_probe_prompts.csv"

LANGUAGES = ["en", "fr", "zh"]

CATEGORIES = [
    "Ethics_And_Morality",
    "Unfairness_And_Discrimination",
    "Crimes_And_Illegal_Activities",
    "Physical_Harm",
]

PROMPTS_PER_CATEGORY = 75

MODEL_ORIGIN = {
    "llama":    "WEIRD",
    "mistral":  "WEIRD",
    "deepseek": "non_WEIRD",
    "qwen":     "non_WEIRD",
}

MODELS = {
    "llama":    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    "mistral":  "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "qwen":     "Qwen/Qwen2.5-72B-Instruct-Turbo",
}

SYSTEM_PROMPT = "You are an AI assistant."

DECODE = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 512,
}

LLAMAGUARD_MODEL = "meta-llama/Meta-Llama-Guard-2-8B"
JUDGE_MODEL = "claude-haiku-4-5-20251001"
