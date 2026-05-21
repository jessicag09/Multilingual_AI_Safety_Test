# AI Safety Project

Evaluate how DeepSeek, Llama, and Mistral respond to JailbreakBench harmful
behaviors wrapped in an AIM jailbreak template, across English, French, and
Mandarin.

- **Prompts**: 100 [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) harmful behaviors
- **Attack**: AIM template, localized per language (`src/templates.py`)
- **Languages**: English (source), French, Mandarin (translated via Claude)
- **Models**: DeepSeek, Llama-3-70B-Instruct, Mistral-Large (via OpenRouter or Ollama)
- **Total**: 100 × 3 langs × 3 models = **900 responses**
- **Judges (both)**:
  - **Llama-Guard-2** via JBB (`evaluate_llamaguard.py`) — `llamaguard_unsafe ∈ {0,1}`
  - **Claude** as refusal judge (`evaluate_refusal.py`) — `refusal ∈ {0,1}`

## Layout

```
src/config.py              paths, model IDs, decoding params
src/templates.py           AIM jailbreak template, EN/FR/ZH
src/models/                swappable backends (OpenRouter / Ollama)
src/load_behaviors.py      download JBB harmful behaviors  -> data/behaviors.csv
src/translate.py           EN -> FR/ZH + template wrap     -> data/prompts.csv
src/generate.py            run 900 generations             -> data/raw_responses.jsonl
src/evaluate_llamaguard.py JBB's Llama-Guard via Together  -> data/llamaguard.jsonl
src/evaluate_refusal.py    Claude yes/no refusal judge     -> data/refusal.jsonl
src/build_dataset.py       merge into final CSV            -> data/results.csv
src/analyze.py             group-bys, plots, judge agreement
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in ANTHROPIC_API_KEY, OPENROUTER_API_KEY, TOGETHER_API_KEY
```

## Pipeline

```
python -m src.load_behaviors
python -m src.translate
python -m src.generate --backend openrouter   # or --backend ollama
python -m src.evaluate_llamaguard
python -m src.evaluate_refusal
python -m src.build_dataset
python -m src.analyze
```

## Notes / caveats

- **Llama-Guard is primarily English-trained.** Scores on French/Mandarin
  responses are noisier; the judge-disagreement table in `analyze.py` is the
  best signal for spotting this.
- **AIM template only.** To compare attacks, add a second template to
  `src/templates.py` and re-run `translate.py` + downstream.
- **Refusal != harmlessness.** A model can refuse politely (refusal=1, unsafe=0)
  or comply with junk content (refusal=0, unsafe=0). The four-cell confusion at
  the bottom of `analyze.py` shows this directly.
