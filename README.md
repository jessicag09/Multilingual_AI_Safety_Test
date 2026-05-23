# Cross-Cultural Safety Evaluation of Open-Source LLMs: A WEIRD-Framed Study

Tests whether the **alignment-decision jurisdiction** of an open-source LLM
(WEIRD vs non-WEIRD) produces measurable differences in safety behavior and
advice framing across response languages, using XSafety's aligned multilingual
prompts plus a custom cultural-norm probe.

Anchored in Henrich's WEIRD framework: AI alignment is developed under
particular institutional/regulatory contexts. Western labs (Meta, OpenAI) make
alignment decisions in SF under US norms; Chinese labs (DeepSeek, Alibaba)
operate under the 2023 Generative AI Measures. This study tests whether those
divergent contexts produce measurably different safety behavior.

## Design

**2 × 2 model origin grid:**

| Origin | Model |
|---|---|
| WEIRD (Meta, US) | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| WEIRD (OpenAI, US) | `openai/gpt-oss-120b` |
| non-WEIRD (DeepSeek, China) | `deepseek-ai/DeepSeek-V4-Pro` |
| non-WEIRD (Alibaba, China) | `Qwen/Qwen3-235B-A22B-Instruct-2507-tput` |

All four served via Together AI serverless.

**2 languages:** en, zh (French dropped after Mistral was removed from the
lineup — Together no longer hosts serverless Mistral and OpenAI's gpt-oss
became the second WEIRD model).

**5 XSafety categories**, chosen for safety stress-testing power + cultural-substrate signal:

| Category | Role |
|---|---|
| `Inquiry_With_Unsafe_Opinion` | Premise-baited questions — tests pushback on harmful framings |
| `Unfairness_And_Discrimination` | Mixed: baited discriminatory statements + value-laden questions |
| `Crimes_And_Illegal_Activities` | Direct first-person criminal requests — tests refusal |
| `Unsafe_Instruction_Topic` | Direct harm-action requests |
| `Mental_Health` | First-person distress prompts — substrate-loaded advice domain |

75 prompts per category × 5 categories × 2 languages = **750 XSafety prompt-language pairs per model**.

**Cultural probe** (`prompts/cultural_probe.json`) — custom non-political stimulus set, archetype-matched:
- **7 archetype-matched pairs** (7 US + 7 CN value-laden cultural-norm prompts)
- Each pair shares life domain, stakes, and discourse act so cross-origin differences reflect substrate framing, not prompt severity
- All translations hardcoded in the JSON (no runtime translator)
- 14 × 2 languages = **28 cultural probe prompt-language pairs per model**
- See `prompts/cultural_probe.json` `_construction_notes` for the Path D
  rationale (political_taboo excluded due to asymmetric regulatory environments)

**Totals:** (750 XSafety + 28 cultural probe) × 4 models = **3,112 generations**.

## Judges (4 — different model families for independence)

| Judge | Model | API | Output | Why this model |
|---|---|---|---|---|
| Harm | `meta-llama/Llama-Guard-4-12B` | Together | `llamaguard_unsafe ∈ {0,1}` | Standard harm classifier |
| Refusal | `claude-haiku-4-5` | Anthropic | `refusal ∈ {0,1}` | No family overlap with eval models |
| XSafety-style | `claude-haiku-4-5` | Anthropic | `xsafety_unsafe ∈ {0,1}` | Same — no family overlap |
| Content/framing | `claude-sonnet-4-6` | Anthropic | structured: stance, **individual_framing (1-5), collectivist_framing (1-5)**, recommendation, tone, framing_notes, refused | Captures substrate signal in advice content; two independent framing axes |

**Why Claude, not gpt-4o-mini?** gpt-4o-mini shares OpenAI's training/safety
substrate with gpt-oss-120b in the eval set — same-family-bias risk. Claude
(Anthropic) has no family overlap with any of the four evaluated models.

**Robustness check:** `evaluate_xsafety_gpt_check.py` re-scores a 250-sample
random subset of XSafety rows with `gpt-4o-mini` using the same XSafety prompt.
Reported as Claude/GPT agreement (raw + Cohen's kappa) in `analyze.py`.

## Layout

```
prompts/cultural_probe.json    14 custom cultural-norm prompts (7 matched pairs), en/zh hardcoded
src/config.py                  paths, MODEL_ORIGIN, categories, judge models
src/models/                    swappable generation backends (Together default)
src/load_xsafety.py            writes XSafety rows -> data/prompts.csv
src/load_cultural_probe.py     appends cultural probe rows -> data/prompts.csv
src/generate.py                runs generations -> data/raw_responses.jsonl  (--max-calls N)
src/evaluate_llamaguard.py     harm classification via Together         -> llamaguard.jsonl
src/evaluate_refusal.py        refusal judge via Claude Haiku           -> refusal.jsonl
src/evaluate_xsafety.py        XSafety-paper-style judge via Claude Haiku -> xsafety_judge.jsonl
src/evaluate_xsafety_gpt_check.py  GPT cross-check on 250 XSafety rows  -> xsafety_gpt_check.jsonl
src/evaluate_content.py        content/framing analyzer via Claude Sonnet -> content_judge.jsonl
src/build_dataset.py           merge all judges -> data/results.csv
src/analyze.py                 RQ1-RQ4 tables, plots, content-judge breakdowns
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in TOGETHER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
```

**API keys required:**
- `TOGETHER_API_KEY` — generation (4 models) + Llama-Guard
- `ANTHROPIC_API_KEY` — refusal, XSafety-style, and content judges
- `OPENAI_API_KEY` — GPT cross-check only (~250 calls)

## Pipeline

```
# Build the prompt set (both stimulus conditions):
python -m src.load_xsafety
python -m src.load_cultural_probe

# Generate responses (resumes if interrupted; --max-calls N to cap a run):
python -m src.generate --backend together

# Score every response with the four judges:
python -m src.evaluate_llamaguard            # Together  ~15 min
python -m src.evaluate_refusal               # Claude    ~30 min
python -m src.evaluate_xsafety               # Claude    ~30 min
python -m src.evaluate_xsafety_gpt_check     # OpenAI    ~3  min  (250 samples)
python -m src.evaluate_content               # Claude    ~45 min  (Sonnet, on all rows)

# Build the merged dataset and run the analysis:
python -m src.build_dataset
python -m src.analyze
```

All judges resume from their own JSONL output files. The full-run judges support
`--max-calls N`; the GPT XSafety cross-check instead uses `--n` to set sample size.

If you only want the XSafety condition, skip `load_cultural_probe`. Analysis
auto-detects which stimulus sets and which judges are present.

## Cost estimate

| Step | Calls | Approx cost |
|---|---|---|
| Generation (4 models × 778 prompt-lang pairs) | 3,112 | ~$4-6 (Together) |
| Llama-Guard judging | 3,112 | ~$0.50 (Together) |
| Claude Haiku refusal | 3,112 | ~$1-2 (Anthropic) |
| Claude Haiku XSafety-style | 3,112 | ~$1-2 (Anthropic) |
| Claude Sonnet content analyzer | 3,112 | ~$6-10 (Anthropic) |
| GPT-4o-mini XSafety cross-check | 250 | ~$0.05 (OpenAI) |
| **Total** | | **~$13-21** |

## Research questions

1. **Replication** — do open-source LLMs degrade in safety on non-English languages, as XSafety found?
2. **Origin × Language interaction** — do Chinese-origin models respond differently in Mandarin than Western-origin models do in their native languages?
3. **Within-origin clustering** — do same-origin models pattern together (origin effect) or diverge (model-specific effect)?
4. **Neutralization vs. cultural loading** — does XSafety's neutralization suppress origin effects, or do they persist (and amplify) on the cultural probe?

## Statistical plan

`analyze.py` outputs cell-mean tables, plots, and CSVs covering:
- Refusal and unsafe rates by model × language × stimulus_set
- WEIRD vs non-WEIRD origin contrasts
- Within-origin clustering tables
- Content-analyzer breakdowns (stance, individual/collectivist framing, recommendation, tone, framing notes)
- Cross-judge agreement (Llama-Guard vs Claude refusal vs Claude XSafety-style)
- Claude/GPT XSafety-judge agreement on the 250-sample robustness check

Inferential tests (two-way ANOVA origin × language; logistic mixed model
origin × language × category fixed, model random) are run at writeup time
from `data/results.csv` in a notebook — not inside `analyze.py`.

## Limitations

- **n=2 per origin cluster** — within-origin clustering is suggestive, not conclusive
- **Cohort confound** — DeepSeek-V4-Pro is newer than Llama-3.3-70B; origin vs training-cohort recency partially confounded
- **Capability-tier matching** is imperfect across model families
- **Llama-Guard-4 is primarily English-trained** — scores on Mandarin responses are noisier; multi-judge design and content analyzer are the cross-checks
- **XSafety has a built-in worldview** — many prompts (especially in Unfairness_And_Discrimination) are constructed such that the "safe" response aligns with a particular value framework. XSafety refusal/safe-rates therefore partially measure agreement with the benchmark's underlying perspective, not pure safety behavior. Our cultural probe is designed to be more value-neutral, and the content analyzer captures framing differences that don't require any particular value framework to score
- **Cultural probe** authored by a single researcher; native-speaker review of translations is the QC step (status: pending)
- **Subsampled XSafety** (75/200 per category, deterministic first-N)
- **Path D scoping** — political_taboo sub-bucket dropped because regulatory environments are asymmetric (China's GAI Measures vs no US parallel); symmetric political probing left to future work
- **No causal attribution** to specific training/regulatory factors — only document patterns
- **No policy recommendation** — descriptive contribution only; normative questions outside this study's scope

## Positioning

Extension of XL-SafetyBench (Anonymous 2026) along three axes:
1. open-source models from culturally-distinct origins, not frontier closed models
2. Mandarin coverage (XL-SafetyBench omits Chinese)
3. explicit WEIRD framing of *alignment-decision jurisdiction* (not training data
   or annotator demographics) to make the cultural-substrate hypothesis falsifiable

The cultural-norm probe sits alongside XSafety as a deliberate culturally-loaded
counterpart. RQ4 contrasts the two conditions directly. The content analyzer
catches substrate signal in advice framing that binary judges miss.
