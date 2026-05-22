# Cross-Cultural Safety Evaluation of Open-Source LLMs: A WEIRD-Framed Study

Tests whether the cultural origin of an open-source LLM produces measurable
differences in its safety behavior across response languages, using XSafety's
aligned multilingual prompts.

Anchored in Henrich's WEIRD framework: safety alignment has been developed
primarily in WEIRD institutional contexts; Chinese-origin models are trained
under different regulatory and corpus conditions. This study tests whether
origin effects appear, persist across languages, and survive a deliberately
neutralized benchmark.

## Design

**2 × 2 model origin grid:**

| Origin | Model |
|---|---|
| WEIRD (Meta, US) | `meta-llama/Meta-Llama-3-70B-Instruct-Turbo` |
| WEIRD (Mistral AI, France) | `mistralai/Mixtral-8x22B-Instruct-v0.1` |
| non-WEIRD (DeepSeek, China) | `deepseek-ai/DeepSeek-V3` |
| non-WEIRD (Alibaba, China) | `Qwen/Qwen2.5-72B-Instruct-Turbo` |

All four served via Together AI. Capability-tier matching is imperfect and
documented as a limitation.

**3 languages:** en, fr, zh — aligned to model origins, professionally
translated by XSafety (no in-pipeline translation).

**4 XSafety categories, each chosen for a theoretical role:**

| Category | Role |
|---|---|
| `Ethics_And_Morality` | Primary WEIRD-effect target — moral norms vary most across cultures |
| `Unfairness_And_Discrimination` | Cultural variation in what counts as discrimination (caste, hukou, laïcité, affirmative action) |
| `Crimes_And_Illegal_Activities` | Legal/regulatory variation across jurisdictions |
| `Physical_Harm` | Universal taboo — control category |

**Stimuli:** 75 prompts per category × 4 categories × 3 languages = 900
prompt-language pairs per model. Identical row indices sampled across
languages so the language axis is isolated from prompt-selection noise.

**Total: 900 × 4 models = 3,600 generations.**

## Cultural probe (custom stimulus set)

Separate from XSafety. Tests whether origin effects persist on
culturally-loaded prompts that XSafety's neutralization removes.

- **45 base prompts**: 15 per origin culture (US / FR / CN)
- Each origin split into two sub-buckets:
  - **`political_taboo`** (5 per origin): Tiananmen, Xinjiang, January 6, abortion, laïcité, etc. — tests political-alignment behavior
  - **`cultural_norm`** (10 per origin): gift-giving, filial piety, lunch length, hukou, etc. — tests cultural-substrate effects on politically-neutral content
- Each base prompt translated to en/fr/zh → 135 prompt-language pairs
- 135 × 4 models = **540 additional generations**

Source file: [`prompts/cultural_probe.json`](prompts/cultural_probe.json).
Loaded via `python -m src.load_cultural_probe` (translates with Claude
Haiku and emits `data/cultural_probe_prompts.csv`).

**Status**: I scaffolded the JSON with all 45 prompts as a starting draft.
**You should review and edit these before running** — they are bias-prone
by their nature, and the original study design calls for native-speaker
review of the translations as well.

## Judges (both run)

| Judge | Output | Why |
|---|---|---|
| **Llama-Guard-2-8B** (via Together) | `llamaguard_unsafe ∈ {0,1}` | Standard harm classifier |
| **Claude Haiku** | `refusal ∈ {0,1}` | Refusal != harmlessness. Also multilingual cross-check for Llama-Guard's English bias |

**Why not Llama-3-70B as judge** (as in the original design): Llama-3 is one
of the four evaluated models. Same-family self-judgment is a confound.

## Layout

```
src/config.py               4 models, 4 categories, n=75, MODEL_ORIGIN map
src/models/                 swappable backends (Together is the default)
src/load_xsafety.py         download XSafety CSVs aligned across en/fr/zh -> data/prompts.csv
src/generate.py             3,600 generations -> data/raw_responses.jsonl  (--max-calls N to cap)
src/evaluate_llamaguard.py  Llama-Guard-2-8B via Together -> data/llamaguard.jsonl
src/evaluate_refusal.py     Claude yes/no refusal judge   -> data/refusal.jsonl
src/build_dataset.py        merge into final CSV          -> data/results.csv
src/analyze.py              origin × language ANOVA tables, within-origin clustering, plots
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in TOGETHER_API_KEY and ANTHROPIC_API_KEY
```

## Pipeline

```
python -m src.load_xsafety
python -m src.generate                          # full run
# or:  python -m src.generate --max-calls 30    # pilot, ~$0.20
python -m src.evaluate_llamaguard
python -m src.evaluate_refusal
python -m src.build_dataset
python -m src.analyze
```

`generate.py` resumes from `raw_responses.jsonl` if interrupted.

## Cost estimate

| Step | Calls | Approx cost |
|---|---|---|
| Generate (4 models × 900 prompt-lang pairs each) | 3,600 | ~$15-25 (Together) |
| Llama-Guard | 3,600 | ~$1 |
| Claude refusal judge | 3,600 | ~$1 |
| **Total** | | **~$17-27** |

## Research questions (pre-registered prediction patterns)

- **RQ1 — Replication**: do open-weight LLMs reproduce XSafety's finding of
  degraded safety in non-English languages?
- **RQ2 — Origin × Language interaction**: do non-WEIRD models behave
  differently on Mandarin than WEIRD models? Does Mistral handle French
  differently than Llama does?
- **RQ3 — Within-origin clustering**: do DeepSeek and Qwen pattern together
  (origin effect) or differ (model-specific effect)?
- **RQ4 — Benchmark neutrality vs. origin**: does XSafety's deliberate
  neutralization suppress origin effects, or do they persist?

The four prediction patterns (origin wins / benchmark wins / mixed by
category / null) are all interpretable, so results are not only confirmatory.

## Statistical plan

- Two-way ANOVA (origin × language), category as repeated factor → main
  effects + interaction
- Within-origin pairwise: DeepSeek vs Qwen; Llama vs Mistral
- Logistic mixed model (origin × language × category fixed, model random)
  is cleaner for the binary outcome — recommended for the final writeup

## Limitations (build into writeup)

- **n=2 per origin** — within-origin clustering is suggestive, not
  conclusive
- **Cohort confound** — DeepSeek-V3 (Dec 2024) is newer than Llama-3
  (Apr 2024); origin-vs-recency is partially confounded
- **Capability-tier matching** is imperfect across families
- **Llama-Guard is primarily English-trained** — judge-disagreement
  analysis is the cross-check
- **No causal attribution** to specific training/regulatory factors; the
  study documents patterns

## Positioning

Extension of XL-SafetyBench (Anonymous 2026) along three axes:
1. open-source models from culturally-distinct origins, not frontier closed models
2. Mandarin coverage (XL-SafetyBench omits Chinese)
3. explicit WEIRD framing to make the cultural-substrate hypothesis falsifiable
