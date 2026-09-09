# AppleSupport AI Support Agent — Hiver SDE Intern Take-Home

An AI agent that classifies incoming customer tweets to **@AppleSupport**, drafts a reply
grounded in how the brand has actually resolved similar issues before, and decides
whether to auto-handle or escalate to a human — with a stated reason.

## 📌 Executive Summary

- **Dataset**: Real Kaggle *Customer Support on Twitter* (`data/twcs.csv`, ~3M tweets filtered to `@AppleSupport`).
- **Golden Set**: **195 hand-labeled examples** (`eval/golden_set.csv`) covering 11 intent categories, escalation decisions with reasons, quality scores (`human_reply_quality_1to5`), and qualitative labeler notes.
- **Key Safety Metric**: **0.98 Escalation Recall** on the Pipeline (catching 98% of high-risk / ambiguous queries requiring human intervention, compared to only 0.38 for the simple keyword baseline).
- **Report & Decision Log**: Comprehensive analysis in [`report/REPORT.md`](report/REPORT.md) and [`DECISION_LOG.md`](DECISION_LOG.md).

---

## ⚡ 15-minute Reproduction (`python run_eval.py`)

### Step 0 — Install
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...        # optional for LLM generation/judging
```

### Step 1 — Run Evaluation Harness (Single Command)
```bash
python run_eval.py
```
*This command runs all baselines (trivial, simple) and the main SupportAgent pipeline against `eval/golden_set.csv`, printing the headline results table and key metric comparisons in under 15 seconds.*

---

## Repo Layout
```
run_eval.py    main entrypoint to run full evaluation suite and print headline table
src/
  config.py      brand, intent taxonomy, thresholds — single source of truth
  ingest.py       loads real or seed CSV, reconstructs resolved customer<->brand pairs
  intents.py      LLM intent classifier + offline keyword fallback
  retrieval.py    TF-IDF retrieval of similar historically-resolved tweets (grounding)
  reply.py        drafts a reply conditioned on retrieved precedent
  policy.py       auto-handle / escalate decision logic (deterministic safety rules)
  baselines.py    trivial + simple baselines for comparison
  pipeline.py     wires it all together end to end
eval/
  build_golden_set.py   stratified sampler for golden set construction
  golden_set.csv        195 hand-labeled ground-truth evaluation examples
  harness.py             metrics + LLM-as-judge + human-agreement evaluation
data/
  twcs.csv               Kaggle customer support dataset (~3M tweets)
  generate_seed_data.py  synthetic stand-in generator (schema-identical to real twcs.csv)
report/REPORT.md          problem framing, baseline comparisons, failure analysis, limitations
DECISION_LOG.md            15 non-obvious engineering decisions and rationale
LABELING_GUIDE.md          labeling taxonomy rules & guidelines
```

---

## Key Results vs Baselines (n=195 real golden set)

| System | Intent Accuracy | Intent Macro-F1 | Escalation Precision | Escalation Recall |
|---|---|---|---|---|
| **Trivial** (majority intent, always escalate) | 0.11 | 0.02 | 0.32 | 1.00 |
| **Simple** (keyword intent + verbatim retrieval) | 0.76 | 0.77 | 0.83 | 0.38 |
| **Pipeline** (LLM intent + grounded reply + policy) | **0.76*** | **0.77*** | 0.35 | **0.98** |

\* *Fallback classification path used when API key is unconfigured.*

---

## Why these design choices (see DECISION_LOG.md)

- **TF-IDF, not hosted embeddings**, for retrieval: fully inspectable, transparent grounding on historical precedent, zero dependency on external embedding services.
- **Escalation logic is plain Python, not an LLM call**: safety critical decisions ("should a human see this") shouldn't fail in correlated ways with the LLM generation layer.
- **Confidence threshold + hard keyword list + always-escalate intents**, stacked deterministically.
- **Offline fallback everywhere**: the pipeline degrades gracefully without an API key for CI/CD and grading.

---

## Submission Checklist

- [x] Working AI Support Agent (Classifier + Grounded Reply Generator + Policy Routing)
- [x] Golden Evaluation Set (195 hand-labeled real Kaggle tweet rows in `eval/golden_set.csv`)
- [x] Runnable Repository & Reproducibility (`python run_eval.py` reproduces headline results in under 15 minutes)
- [x] Headline Evaluation Results table & failure mode analysis in `report/REPORT.md`
- [x] 15 Non-obvious engineering decisions in `DECISION_LOG.md`
