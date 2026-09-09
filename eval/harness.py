"""
Runs the pipeline (or a baseline) over the golden set and reports:

1. Intent accuracy / per-class precision-recall vs human_intent_label.
2. Escalation decision precision/recall vs human_should_escalate.
3. LLM-as-judge score (1-5) for each drafted reply on 4 rubric dimensions.
4. Agreement between the LLM judge and a human rating on a subsample —
   this is NOT optional; a judge nobody has checked against a human is
   just a second unverified model.

Usage:
    python -m eval.harness --system pipeline   # our system
    python -m eval.harness --system trivial
    python -m eval.harness --system simple
    python -m eval.harness --judge-agreement    # requires human_reply_quality_1to5 filled in
"""
import argparse
import json
import re
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import classification_report, precision_recall_fscore_support

from src.config import GOLDEN_SET_PATH, JUDGE_MODEL, INTENTS
from src.ingest import load_raw, build_pairs
from src.pipeline import SupportAgent
from src.baselines import trivial_baseline, simple_baseline, majority_intent
from src.llm_client import complete, USING_LLM

JUDGE_SYSTEM = (
    "You are grading a customer-support Twitter reply. Score it 1-5 on each dimension. "
    "Respond ONLY with JSON: {\"groundedness\": <1-5>, \"correctness\": <1-5>, "
    "\"tone\": <1-5>, \"actionability\": <1-5>, \"overall\": <1-5>, \"why\": \"<one sentence>\"}. "
    "groundedness = does it match how this brand actually resolves similar issues, "
    "without inventing policy. correctness = is the advice actually right/safe. "
    "tone = brand-appropriate, empathetic, not robotic. actionability = customer knows "
    "exactly what to do next."
)


def judge_reply(customer_text: str, draft_reply: str, grounding_example: str) -> dict:
    prompt = (
        f"Customer message: \"{customer_text}\"\n"
        f"Historical precedent this should be grounded in: \"{grounding_example}\"\n"
        f"Drafted reply being graded: \"{draft_reply}\""
    )
    raw = complete(JUDGE_SYSTEM, prompt, model=JUDGE_MODEL, max_tokens=150)
    try:
        return json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
    except Exception:
        return {"groundedness": None, "correctness": None, "tone": None,
                "actionability": None, "overall": None, "why": "judge output unparseable"}


def load_golden():
    try:
        gs = pd.read_csv(GOLDEN_SET_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{GOLDEN_SET_PATH} not found. Run `python -m eval.build_golden_set`, "
            "hand-label eval/golden_set_TO_LABEL.csv, then save it as "
            f"{GOLDEN_SET_PATH}."
        )
    missing = gs["human_intent_label"].isna() | (gs["human_intent_label"] == "")
    if missing.all():
        raise ValueError("golden_set.csv has no human labels filled in yet.")
    if missing.any():
        print(f"WARNING: {missing.sum()} rows have no human_intent_label — dropping them for this run.")
        gs = gs[~missing]
    return gs


def get_predictor(system: str):
    df, is_real = load_raw()
    pairs = build_pairs(df)
    if system == "pipeline":
        agent = SupportAgent()
        return lambda text: agent.handle(text), is_real
    elif system == "trivial":
        maj = majority_intent(pairs)
        return trivial_baseline(pairs, maj), is_real
    elif system == "simple":
        return simple_baseline(pairs), is_real
    raise ValueError(system)


def run(system: str, judge: bool, judge_sample_n: int):
    gs = load_golden()
    predict, is_real = get_predictor(system)
    if not is_real:
        print("*** WARNING: evaluating against SYNTHETIC seed data. These numbers are "
              "placeholders — see report/REPORT.md 'misleading headline number' section. ***")

    rows = []
    for _, r in gs.iterrows():
        pred = predict(r["customer_text"])
        pred_intent = pred["intent"] if isinstance(pred, dict) else pred["intent"]
        pred_action = pred.get("action")
        pred_reply = pred.get("reply") or pred.get("draft_reply")
        rows.append({
            "customer_text": r["customer_text"],
            "true_intent": r["human_intent_label"],
            "pred_intent": pred_intent,
            "true_escalate": str(r["human_should_escalate"]).strip().lower() in ("yes", "true", "1"),
            "pred_escalate": pred_action != "auto_handle",
            "pred_reply": pred_reply,
            "human_quality": r.get("human_reply_quality_1to5"),
        })
    results = pd.DataFrame(rows)

    print("\n=== INTENT CLASSIFICATION ===")
    valid = results[results["true_intent"].isin(INTENTS)]
    print(classification_report(valid["true_intent"], valid["pred_intent"], zero_division=0))

    print("=== ESCALATION DECISION ===")
    p, r_, f1, _ = precision_recall_fscore_support(
        results["true_escalate"], results["pred_escalate"], average="binary", zero_division=0
    )
    print(f"precision={p:.2f} recall={r_:.2f} f1={f1:.2f}  "
          f"(recall matters more here — a missed escalation reaches a customer unsupervised)")

    if judge and system == "pipeline":
        print(f"\n=== LLM-AS-JUDGE (sampling {judge_sample_n} replies) ===")
        if not USING_LLM:
            print("No ANTHROPIC_API_KEY set — skipping judge (would just grade stub text).")
        else:
            sample = results.sample(min(judge_sample_n, len(results)), random_state=1)
            scores = []
            for _, row in sample.iterrows():
                j = judge_reply(row["customer_text"], row["pred_reply"], row["customer_text"])
                scores.append(j)
            jdf = pd.DataFrame(scores)
            print(jdf[["groundedness", "correctness", "tone", "actionability", "overall"]].mean())
            jdf.to_csv("eval/judge_scores_latest.csv", index=False)
            print("Saved per-row judge scores to eval/judge_scores_latest.csv")

    results.to_csv(f"eval/results_{system}.csv", index=False)
    print(f"\nSaved full results to eval/results_{system}.csv")
    return results


def judge_agreement():
    """
    Correlates LLM judge 'overall' score against the human's
    human_reply_quality_1to5 rating on whatever rows have BOTH filled in.
    This is the evidence the brief requires for judge trustworthiness.
    """
    gs = pd.read_csv(GOLDEN_SET_PATH)
    labeled = gs.dropna(subset=["human_reply_quality_1to5"])
    labeled = labeled[labeled["human_reply_quality_1to5"] != ""]
    if len(labeled) < 10:
        print(f"Only {len(labeled)} rows have a human_reply_quality_1to5 rating. "
              "Need at least ~20-30 for a meaningful correlation — label more rows.")
        return
    if not USING_LLM:
        print("No ANTHROPIC_API_KEY set — cannot compute judge scores for agreement check.")
        return
    judge_scores = []
    for _, row in labeled.iterrows():
        j = judge_reply(row["customer_text"], row.get("support_text", ""), row.get("support_text", ""))
        judge_scores.append(j.get("overall"))
    labeled = labeled.assign(judge_overall=judge_scores).dropna(subset=["judge_overall"])
    r_val, p_val = pearsonr(labeled["human_reply_quality_1to5"].astype(float), labeled["judge_overall"].astype(float))
    print(f"Pearson r between human and LLM-judge scores: {r_val:.2f} (p={p_val:.3f}), n={len(labeled)}")
    print("Rule of thumb used in report: r > 0.6 = judge is trustworthy enough to scale eval with; "
          "below that, treat every judge number as provisional and keep sampling humans.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["pipeline", "trivial", "simple"], default="pipeline")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge_sample_n", type=int, default=30)
    ap.add_argument("--judge-agreement", dest="judge_agreement", action="store_true")
    args = ap.parse_args()
    if args.judge_agreement:
        judge_agreement()
    else:
        run(args.system, args.judge, args.judge_sample_n)
