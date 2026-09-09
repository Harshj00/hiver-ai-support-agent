"""
Grounding retrieval: given an incoming customer message, find the K most
similar HISTORICALLY RESOLVED customer messages to this brand, and return
how the brand actually replied to those. This is what "grounded in how
the brand has historically resolved similar issues" means concretely —
we are not asking the LLM to invent policy, we're asking it to imitate
precedent.

Decision (DECISION_LOG.md #4): TF-IDF, not embeddings from a hosted model.
This environment has no access to embedding-model weight hosts, and more
importantly, TF-IDF is fully inspectable — for a support agent that will
be trusted with real customers, "why did it say that" traces back to
exact past tweets, not an opaque vector.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.ingest import flag_dm_redirect


class ReplyRetriever:
    def __init__(self, pairs_df: pd.DataFrame):
        self.pairs = pairs_df.reset_index(drop=True)
        # Decision (DECISION_LOG.md #16): rank substantive resolutions above
        # generic "DM us" redirects. ~33% of real AppleSupport replies are
        # pure redirects (see src/ingest.py flag_dm_redirect) — grounding on
        # those teaches the agent to say "DM us" for everything, which looks
        # fine on similarity but adds little value. We still allow redirects
        # to surface when nothing more substantive is close enough.
        self.pairs["_is_redirect"] = self.pairs["support_text"].apply(flag_dm_redirect)

        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.9
        )
        self.matrix = self.vectorizer.fit_transform(self.pairs["customer_text"])

    def top_k(self, query: str, k: int = 3, redirect_penalty: float = 0.15) -> pd.DataFrame:
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix)[0].copy()
        # soft penalty, not a hard filter — a highly similar redirect can still
        # beat a barely-similar substantive reply, which is the right call
        sims = sims - (self.pairs["_is_redirect"].values * redirect_penalty)
        top_idx = sims.argsort()[::-1][:k]
        result = self.pairs.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result[["customer_text", "support_text", "similarity", "_is_redirect"]].rename(
            columns={"_is_redirect": "is_redirect_example"}
        )