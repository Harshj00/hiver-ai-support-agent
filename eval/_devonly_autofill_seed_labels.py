"""
DEV-ONLY. Not a deliverable, not real labeling.

Auto-fills eval/golden_set_TO_LABEL.csv using the intent_true/resolved_true
columns that only exist in our synthetic seed data, purely so harness.py's
code paths (metrics, judge, agreement) can be smoke-tested before you do
the real hand-labeling against real data.

DO NOT submit numbers produced this way. The report has a section
specifically about this — see report/REPORT.md.
"""
import pandas as pd
from src.config import GOLDEN_SAMPLE_PATH, GOLDEN_SET_PATH
from src.ingest import load_raw, build_pairs

raw, is_real = load_raw()
assert not is_real, "This script should only ever touch the synthetic seed."

df = pd.read_csv(GOLDEN_SAMPLE_PATH)
seed = pd.read_csv("data/seed_conversations.csv")
truth = seed.set_index("tweet_id")[["intent_true", "resolved_true"]]

df["human_intent_label"] = df["customer_tweet_id"].map(truth["intent_true"])
# simulate a human deciding to escalate on 'other' intent or low-signal cases
df["human_should_escalate"] = df["human_intent_label"].apply(
    lambda i: "yes" if i == "other" else "no"
)
df["human_escalate_reason"] = "auto-simulated for smoke test"
# fake a plausible quality distribution for the judge-agreement smoke test
import random
random.seed(3)
df["human_reply_quality_1to5"] = [random.choice([3, 4, 4, 5, 5]) for _ in range(len(df))]
df["labeler_notes"] = "SMOKE TEST ONLY — not a real human label"

df.to_csv(GOLDEN_SET_PATH, index=False)
print(f"Wrote SMOKE-TEST-ONLY labels to {GOLDEN_SET_PATH} ({len(df)} rows)")
