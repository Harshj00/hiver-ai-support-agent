"""
Builds eval/golden_set_TO_LABEL.csv — a stratified sample for YOU to
hand-label. This script does the sampling; it does NOT invent labels.

Sampling method (documented here because the brief asks how you sampled):
1. Take all resolved customer->brand pairs.
2. Run the keyword fallback classifier (src.intents._fallback_classify) to
   get a *rough* intent bucket per message — this is just for stratification,
   not the ground truth.
3. Sample proportionally across buckets, with a floor of `min_per_bucket`
   so rare intents aren't sampling zero, and a deliberate 15% "unstratified"
   random top-up so intents you didn't anticipate still have a chance to
   show up in the golden set (see DECISION_LOG.md #2).

Usage:
    python -m eval.build_golden_set --n 200
"""
import argparse
import pandas as pd
from src.ingest import load_raw, build_pairs
from src.intents import _fallback_classify
from src.config import GOLDEN_SAMPLE_PATH


def main(n: int, min_per_bucket: int, seed: int):
    df, is_real = load_raw()
    pairs = build_pairs(df)
    if not is_real:
        print("WARNING: sampling from SYNTHETIC seed data. Re-run this against "
              "data/twcs.csv before labeling for real — labels on fake data don't count.")

    pairs["rough_bucket"] = pairs["customer_text"].apply(lambda t: _fallback_classify(t)["intent"])

    n_random_topup = int(n * 0.15)
    n_stratified = n - n_random_topup

    buckets = pairs.groupby("rough_bucket")
    per_bucket = max(min_per_bucket, n_stratified // buckets.ngroups)
    stratified_samples = []
    for _, group in buckets:
        take = min(len(group), per_bucket)
        stratified_samples.append(group.sample(take, random_state=seed))
    stratified = pd.concat(stratified_samples)

    remaining = pairs.drop(stratified.index)
    topup = remaining.sample(min(n_random_topup, len(remaining)), random_state=seed)

    final = pd.concat([stratified, topup]).drop_duplicates(subset=["customer_tweet_id"])
    final = final.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle so labeling isn't bucket-ordered

    final["human_intent_label"] = ""       # YOU fill this in — pick from src/config.py INTENTS
    final["human_should_escalate"] = ""    # YOU fill this in — yes/no + why, in the next column
    final["human_escalate_reason"] = ""
    final["human_reply_quality_1to5"] = "" # YOU fill this in after running the pipeline on this set
    final["labeler_notes"] = ""

    out_cols = ["customer_tweet_id", "customer_text", "support_text", "rough_bucket",
                "human_intent_label", "human_should_escalate", "human_escalate_reason",
                "human_reply_quality_1to5", "labeler_notes"]
    final[out_cols].to_csv(GOLDEN_SAMPLE_PATH, index=False)
    print(f"Wrote {len(final)} rows to {GOLDEN_SAMPLE_PATH}")
    print("Bucket distribution (rough, pre-labeling):")
    print(final["rough_bucket"].value_counts())
    print("\nNEXT STEP (you, by hand): open this CSV and fill in the human_* columns.")
    print("Budget ~2-3 min/row. Rename the file to eval/golden_set.csv when done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min_per_bucket", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n, args.min_per_bucket, args.seed)
