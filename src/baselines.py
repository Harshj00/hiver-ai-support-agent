"""
Two baselines the report compares against, as required by the brief.

TRIVIAL: majority-class intent, one canned reply, always escalate.
    This is the floor — if our system can't beat this, it's worthless.

SIMPLE: keyword-based intent (src.intents._fallback_classify), reply =
    verbatim closest historical reply via TF-IDF retrieval (no LLM drafting),
    escalate = fixed keyword list only.
    This isolates "how much does the LLM actually add" over cheap retrieval.
"""
from collections import Counter
from src.intents import _fallback_classify
from src.retrieval import ReplyRetriever


def trivial_baseline(pairs_df, majority_intent: str):
    def predict(customer_text: str):
        return {
            "intent": majority_intent,
            "reply": "Thanks for reaching out! A member of our support team will follow up with you shortly.",
            "action": "escalate",
            "action_reason": "trivial baseline always escalates",
        }
    return predict


def simple_baseline(pairs_df):
    retriever = ReplyRetriever(pairs_df)

    def predict(customer_text: str):
        intent_result = _fallback_classify(customer_text)
        top = retriever.top_k(customer_text, k=1).iloc[0]
        action = "escalate" if intent_result["confidence"] < 0.6 else "auto_handle"
        return {
            "intent": intent_result["intent"],
            "reply": top["support_text"],
            "action": action,
            "action_reason": f"keyword confidence {intent_result['confidence']:.2f}",
        }
    return predict


def majority_intent(pairs_df, intent_col="intent_true") -> str:
    if intent_col not in pairs_df.columns:
        return "other"
    return Counter(pairs_df[intent_col]).most_common(1)[0][0]
