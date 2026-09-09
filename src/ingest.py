"""
Loads the raw twitter customer-support CSV (real Kaggle file if present,
otherwise the synthetic seed) and reconstructs (customer_tweet -> support_reply)
pairs for one brand.

Real twcs.csv has NO intent or resolved labels — those only exist in our
seed file for self-checking. Downstream code must never assume intent_true
/ resolved_true exist; ingest.py strips them for anything claiming to
come from "real" data flow.
"""
import os
import re
import pandas as pd
from src.config import RAW_CSV_PATH, SEED_CSV_PATH, BRAND

_MENTION_ID_RE = re.compile(r"@\d+")          # anonymized customer ids, e.g. @115854
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")   # @AppleSupport and similar handles
_URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """
    Strip the Twitter-specific noise that has nothing to do with the
    customer's actual issue: anonymized reply-target ids (@115854), brand/
    user handles (@AppleSupport), and t.co tracking links. Without this,
    TF-IDF retrieval and the intent classifier partly match on "did this
    tweet also get redirected to a DM link" rather than the issue itself.
    """
    if not isinstance(text, str):
        return ""
    t = _URL_RE.sub("", text)
    t = _MENTION_ID_RE.sub("", t)
    t = _MENTION_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def load_raw() -> pd.DataFrame:
    if os.path.exists(RAW_CSV_PATH):
        print(f"[ingest] Using REAL data: {RAW_CSV_PATH}")
        df = pd.read_csv(RAW_CSV_PATH, dtype=str)
        is_real = True
    elif os.path.exists(SEED_CSV_PATH):
        print(f"[ingest] WARNING: real data not found at {RAW_CSV_PATH}.")
        print(f"[ingest] Falling back to SYNTHETIC seed data: {SEED_CSV_PATH}")
        print("[ingest] Every downstream number is a placeholder until you swap in the real file.")
        df = pd.read_csv(SEED_CSV_PATH, dtype=str)
        is_real = False
    else:
        raise FileNotFoundError(
            "No data found. Run `python data/generate_seed_data.py` for a quick "
            f"smoke test, or place the real Kaggle file at {RAW_CSV_PATH}."
        )
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"
    return df, is_real


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct one row per (customer tweet, brand's reply to it), restricted
    to threads where the brand is the responder.

    Implementation note: this used to iterate row-by-row over EVERY customer
    tweet in the dataset (all brands, ~1.5M rows on the real file) calling
    .iterrows() — fine on 390 seed rows, unusably slow at real scale. Fixed
    by first filtering to just this brand's replies (a few thousand rows for
    most brands) and doing a single vectorized merge on
    in_response_to_tweet_id, instead of looping over everything else's
    traffic to find them.
    """
    brand_replies = df[(~df["inbound"]) & (df["author_id"] == BRAND)].copy()
    brand_replies = brand_replies[brand_replies["in_response_to_tweet_id"].notna() &
                                   (brand_replies["in_response_to_tweet_id"] != "")]

    customer_tweets = df[df["inbound"]][["tweet_id", "text", "created_at"]].copy()
    customer_tweets = customer_tweets.rename(columns={"text": "customer_text"})

    merged = brand_replies.merge(
        customer_tweets,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_support", "_customer"),
    )

    pairs = merged.rename(columns={
        "tweet_id_customer": "customer_tweet_id",
        "text": "support_text",
        "created_at_customer": "created_at",
    })[["customer_tweet_id", "customer_text", "support_text", "created_at"]]

    pairs = pairs.drop_duplicates(subset=["customer_tweet_id"]).reset_index(drop=True)

    # keep raw text for audit/citation, use cleaned text everywhere downstream
    # (retrieval, classification, LLM prompts) — see clean_text() above.
    pairs["customer_text_raw"] = pairs["customer_text"]
    pairs["support_text_raw"] = pairs["support_text"]
    pairs["customer_text"] = pairs["customer_text"].apply(clean_text)
    pairs["support_text"] = pairs["support_text"].apply(clean_text)
    pairs = pairs[(pairs["customer_text"] != "") & (pairs["support_text"] != "")]

    return pairs.reset_index(drop=True)


def flag_dm_redirect(support_text: str) -> bool:
    """
    Heuristic: does this 'resolution' actually resolve anything, or is it
    just a template redirecting the customer to DM? This matters a lot —
    if most of your grounding examples are redirects, your agent will
    learn to say 'DM us' for everything and look well-grounded while
    adding little value. See report/REPORT.md.
    """
    t = support_text.lower()
    redirect_phrases = [
        "join us in a dm", "go to dm", "send us a dm", "dm us",
        "direct message", "meet us in dm", "let's dm",
    ]
    return any(p in t for p in redirect_phrases)


if __name__ == "__main__":
    df, is_real = load_raw()
    pairs = build_pairs(df)
    print(f"[ingest] {len(pairs)} resolved customer->{BRAND} pairs found "
          f"({'REAL' if is_real else 'SYNTHETIC'} data).")
    print(pairs[["customer_text", "support_text"]].head(3).to_string())

    print("\n[ingest] --- Quick diagnostic: how many 'resolutions' are just DM redirects? ---")
    n_redirect = pairs["support_text"].apply(flag_dm_redirect).sum()
    pct = 100 * n_redirect / len(pairs) if len(pairs) else 0
    print(f"[ingest] {n_redirect}/{len(pairs)} ({pct:.1f}%) of support replies are DM-redirect "
          f"templates, not substantive resolutions.")
    if pct > 40:
        print("[ingest] That's a large fraction. Worth its own line in report/REPORT.md's "
              "'misleading headline number' section, and possibly its own policy: route "
              "high-similarity-to-redirect-only cases differently than genuinely resolved ones.")