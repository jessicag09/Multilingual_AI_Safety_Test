CROSS-CULTURAL SAFETY EVALUATION OF OPEN-WEIGHT LLMS: A WEIRD-FRAMED STUDY
==========================================================================


This project tests whether the alignment-decision jurisdiction of an
open-weight LLM (WEIRD vs non_WEIRD) is associated with measurable
differences in:

- baseline safety behavior on XSafety prompts
- adolescent-safety behavior on a custom cross-cultural benchmark
- advice framing across languages and model families

The study treats model origin as a theoretical lens, not a causal label.
Western labs (Meta, OpenAI) and Chinese labs (DeepSeek, Alibaba) make
alignment decisions under different institutional, regulatory, and cultural
conditions. The goal here is to measure response patterns, not to prove that
any one factor caused them.

Design
------


Model grid
----------


Origin | Model key | Model
WEIRD | llama | meta-llama/Llama-3.3-70B-Instruct-Turbo
WEIRD | gpt_oss | openai/gpt-oss-120b
non_WEIRD | deepseek | deepseek-ai/DeepSeek-V4-Pro
non_WEIRD | qwen | Qwen/Qwen3-235B-A22B-Instruct-2507-tput

All four are served through Together AI serverless.

Languages
---------


- en
- zh

Stimulus sets
-------------


1. XSafety baseline
-------------------


Five XSafety categories are used as the neutralized benchmark condition:

Category | Role in study
Inquiry_With_Unsafe_Opinion | premise-baited questions; tests pushback on harmful framings
Unfairness_And_Discrimination | discriminatory or value-laden prompts
Crimes_And_Illegal_Activities | direct criminal requests; primarily refusal/compliance
Unsafe_Instruction_Topic | direct dangerous how-to requests
Mental_Health | first-person distress prompts; advice-heavy safety domain

Sampling:

- 75 prompts per category
- 5 categories
- 2 languages
- 750 XSafety prompt-language pairs per model

2. Cultural probe
-----------------


prompts/cultural_probe.json is the custom adolescent vulnerable-user
benchmark. It contains 32 base prompts:

- 12 archetype-matched US/CN pairs (24 prompts total)
- 8 culture-specific supplemental prompts with no direct pair

Role split:

- 16 core
- 8 control
- 8 supplemental_unpaired

Prompt-language expansion:

- 32 base prompts
- 64 prompt-language pairs per model

Topic coverage includes:

- academic crisis and despair
- self-harm concealment
- parental physical abuse
- disordered eating
- substance pressure
- grooming / older strangers
- LGBTQ family rejection
- sextortion
- caregiving burden
- cosmetic surgery pressure
- plus country-specific supplementals such as left_behind_child,
  internet_addiction_camp_fear, firearm_access_home_si, and
  vocational_track_streaming

The benchmark is intentionally non-political. Political taboo prompts are
excluded because the Chinese and US regulatory environments are asymmetrical
enough that such prompts would dominate the signal.

Total generation count
----------------------


Per model:

- 750 XSafety prompt-language pairs
- 64 cultural probe prompt-language pairs
- 814 total rows per model

Across 4 models:

- 3256 total generations

Judges
------


The evaluation stack is intentionally split into baseline safety, adolescent
safety, and framing.

Primary judges
--------------


Judge | Script | Model | Scope | Main output
Harm classifier | src/evaluate_llamaguard.py | meta-llama/Llama-Guard-4-12B | all rows | llamaguard_unsafe
Refusal judge | src/evaluate_refusal.py | claude-haiku-4-5-20251001 | all rows | refusal
XSafety-style judge | src/evaluate_xsafety.py | claude-haiku-4-5-20251001 | all rows | xsafety_unsafe
Adolescent safety judge | src/evaluate_content.py | claude-sonnet-4-6 | cultural_probe by default | overall_safe, facilitates_harm, risk_level, etc.
Framing judge | src/evaluate_framing.py | claude-sonnet-4-6 | all rows by default | stance, individual_framing, collectivist_framing, recommendation, tone

Cross-judge robustness checks
-----------------------------


Check | Script | Model | Default scope
XSafety check | src/evaluate_xsafety_gpt_check.py | gpt-4o-mini | 250 sampled XSafety rows
Content check | src/evaluate_content_check.py | gpt-4o-mini | 100 sampled cultural_probe rows
Framing check | src/evaluate_framing_check.py | gpt-4o-mini | 150 sampled rows from the primary framing slice

The primary framing slice is:

- all cultural_probe rows
- plus advice-style XSafety categories:
  - Mental_Health
  - Inquiry_With_Unsafe_Opinion
  - Unfairness_And_Discrimination

Why Claude is the main judge
----------------------------


The evaluated models already include Meta/Llama, OpenAI, DeepSeek, and Qwen.
Using Anthropic as the primary evaluator reduces same-family bias in the main
judge stack. OpenAI is used only as a secondary robustness check.

Evaluation logic
----------------


Safety and framing are separate on purpose
------------------------------------------


The project treats these as different questions:

- safety: did the model avoid making the situation worse?
- framing: how did the model present its advice?

A response can be safe but framed differently across languages or model
families. Keeping them separate prevents style differences from being confused
with safety failures.

Adolescent safety judge
-----------------------


evaluate_content.py is the main benchmark-specific evaluator. It is designed
for vulnerable minors and defaults to cultural_probe only.

It scores:

- risk_level
- recognizes_risk
- facilitates_harm
- appropriate_escalation
- supportive_tone
- protective_guidance
- refused
- overall_safe
- failure_notes

The rubric first distinguishes:

- distress / help-seeking
- harmful-request
- normalization probe

That matters because a hard refusal is correct for some harmful requests but is
unsafe for a distress prompt.

Framing judge
-------------


evaluate_framing.py is the secondary descriptive lens. It defaults to all
rows and scores:

- stance
- individual_framing (1-5)
- collectivist_framing (1-5)
- recommendation
- tone
- refused
- framing_notes

The two-axis design allows responses to score high on both individual and
collectivist framing, which a single scale would miss.

Repository layout
-----------------



CODE
------------------------------------------------------------------------
prompts/cultural_probe.json         adolescent cultural probe source
src/config.py                       paths, model config, category lists
src/load_xsafety.py                 writes XSafety rows -> data/prompts.csv
src/load_cultural_probe.py          writes cultural probe rows -> data/prompts.csv
src/generate.py                     runs generations -> data/raw_responses.jsonl
src/evaluate_llamaguard.py          Llama-Guard baseline harm classifier
src/evaluate_refusal.py             Claude Haiku refusal judge
src/evaluate_xsafety.py             Claude Haiku XSafety-style judge
src/evaluate_content.py             Claude Sonnet adolescent safety judge
src/evaluate_framing.py             Claude Sonnet framing judge
src/evaluate_xsafety_gpt_check.py   OpenAI robustness check for XSafety labels
src/evaluate_content_check.py       OpenAI robustness check for adolescent safety labels
src/evaluate_framing_check.py       OpenAI robustness check for framing labels
src/build_dataset.py                merges prompt meta + judge outputs -> data/results.csv
src/analyze.py                      summary tables, plots, and robustness summaries
------------------------------------------------------------------------


Setup
-----



CODE
------------------------------------------------------------------------
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
------------------------------------------------------------------------


Fill in:

- TOGETHER_API_KEY
- ANTHROPIC_API_KEY
- OPENAI_API_KEY

API usage
---------


- TOGETHER_API_KEY: model generation + Llama-Guard
- ANTHROPIC_API_KEY: refusal, XSafety-style, content, and framing judges
- OPENAI_API_KEY: cross-judge robustness checks only

Pipeline
--------


1. Build prompt table
---------------------



CODE
------------------------------------------------------------------------
python -m src.load_xsafety
python -m src.load_cultural_probe
------------------------------------------------------------------------


data/prompts.csv is the prompt-table source of truth used by generation and
downstream prompt metadata merges.

2. Generate responses
---------------------



CODE
------------------------------------------------------------------------
python -m src.generate --backend together
------------------------------------------------------------------------


Generation is append-only and resume-safe.

3. Run the judge stack
----------------------



CODE
------------------------------------------------------------------------
python -m src.evaluate_llamaguard
python -m src.evaluate_refusal
python -m src.evaluate_xsafety
python -m src.evaluate_xsafety_gpt_check
python -m src.evaluate_content
python -m src.evaluate_framing
python -m src.evaluate_content_check
python -m src.evaluate_framing_check
------------------------------------------------------------------------


Notes:

- evaluate_content defaults to --stimulus-set cultural_probe
- evaluate_framing defaults to --stimulus-set all
- all full-run judges support --max-calls N
- sampled GPT checks use --n

4. Merge outputs and analyze
----------------------------



CODE
------------------------------------------------------------------------
python -m src.build_dataset
python -m src.analyze
------------------------------------------------------------------------


If matplotlib cache permissions are noisy in your shell, run:


CODE
------------------------------------------------------------------------
mkdir -p /tmp/mpl
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp python -m src.analyze
------------------------------------------------------------------------


Build behavior and duplicate protection
---------------------------------------


Judge output files are append-only JSONL logs. If a judge is accidentally
launched twice, duplicate key rows can appear in the raw audit files.

src/build_dataset.py protects the merged dataset by:

- deduping each judge file on
  model, stimulus_set, prompt_id, language
- keeping the latest row
- excluding raw_judge_output from data/results.csv

This makes results.csv robust even if an audit log is messy.

Analysis outputs
----------------


src/analyze.py reads data/results.csv and writes summary CSVs and plots.

Main sections:

1. Baseline safety on XSafety
   - llamaguard_unsafe
   - refusal
   - xsafety_unsafe
2. Cultural probe baseline contrasts
   - model origin vs prompt origin
   - XSafety vs cultural probe deltas
3. Adolescent safety
   - overall_safe
   - facilitates_harm
   - recognizes_risk
   - appropriate_escalation
   - supportive_tone
   - protective_guidance
4. Framing
   - stance
   - individual_framing
   - collectivist_framing
   - recommendation
   - tone
5. Cross-judge robustness
   - Claude vs GPT on XSafety sampled rows
   - Claude vs GPT on adolescent safety sampled rows
   - Claude vs GPT on framing sampled rows

Important cultural_probe interpretation:

- core pairs: main evidence
- control pairs: robustness / null check
- supplemental_unpaired: descriptive only

Cost estimate
-------------


Rough order-of-magnitude estimate for the full current pipeline:

Step | Approximate cost
Generation on Together | low single-digit USD
Llama-Guard baseline | under $1
Claude Haiku refusal + XSafety-style | low single-digit USD
Claude Sonnet content + framing | mid-to-high teens USD combined
OpenAI GPT cross-checks | well under $1 total

The expensive step is evaluate_framing.py, because it runs Sonnet across all
3256 rows.

Research questions
------------------


1. Do open-weight LLMs exhibit language-dependent safety differences on the
   XSafety benchmark?
2. Do cross-language / cross-model-origin differences extend to vulnerable
   adolescent prompts?
3. When responses are safe or unsafe, do advice-framing patterns differ across
   languages and model families?
4. Do culturally grounded adolescent prompts reveal differences that are weaker
   or hidden on neutralized benchmark prompts?

Limitations
-----------


- n=2 models per origin cluster
- model-origin effects are not cleanly separable from model-family and cohort effects
- the adolescent benchmark is custom and exploratory
- Chinese prompts ideally receive native-speaker review before strong claims
- the safety and framing judges are still LLM judges, not human gold labels
- OpenAI robustness checks are helpful but are not neutral ground truth
- supplemental_unpaired prompts are descriptive, not pairwise causal evidence
- no causal attribution to any specific training, policy, or regulatory mechanism

Current study posture
---------------------


This is best treated as an exploratory benchmark / measurement study:

- strong enough to surface patterns worth reporting
- not strong enough to support sweeping causal claims about culture

The intended contribution is empirical:

- a cleaned XSafety baseline
- a custom adolescent vulnerable-user benchmark
- a safety/framing split that makes the analysis more interpretable
