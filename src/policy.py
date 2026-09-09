"""
Auto-handle vs escalate decision.

Decision (DECISION_LOG.md #6): this logic is plain Python, not an LLM
call. A system that decides "should a human see this" should not itself
be the thing that might be wrong in the same way twice. Hard-coded
signals and thresholds are boring on purpose.
"""
from src.config import HARD_ESCALATE_SIGNALS, AUTO_HANDLE_CONFIDENCE_THRESHOLD, MIN_GROUNDING_SIMILARITY

ALWAYS_ESCALATE_INTENTS = {"other", "security_phishing"}
# security_phishing always escalates: telling a customer "this isn't phishing"
# and being wrong is a much worse failure than a slower human confirmation.

# Billing disputes are auto-drafted for the customer-service tone, but never
# auto-SENT — money and warranty language creates real liability if wrong.
REVIEW_REQUIRED_INTENTS = {"warranty_billing"}


def decide(intent: str, confidence: float, customer_text: str, top_similarity: float = 1.0) -> dict:
    text_lower = customer_text.lower()

    hit = next((s for s in HARD_ESCALATE_SIGNALS if s in text_lower), None)
    if hit:
        return {"action": "escalate", "reason": f"hard-escalate signal matched: '{hit}'"}

    if intent in ALWAYS_ESCALATE_INTENTS:
        return {"action": "escalate", "reason": f"intent '{intent}' is always routed to a human"}

    if confidence < AUTO_HANDLE_CONFIDENCE_THRESHOLD:
        return {"action": "escalate",
                "reason": f"classifier confidence {confidence:.2f} below threshold "
                          f"{AUTO_HANDLE_CONFIDENCE_THRESHOLD}"}

    # A confident intent label says nothing about whether we found a GOOD precedent
    # to ground the reply on. These are two different failure modes and both need
    # their own check — see DECISION_LOG.md #17.
    if top_similarity < MIN_GROUNDING_SIMILARITY:
        return {"action": "escalate",
                "reason": f"weak grounding: closest historical precedent similarity "
                          f"{top_similarity:.2f} below threshold {MIN_GROUNDING_SIMILARITY} "
                          f"— confident about the intent, not about the precedent"}

    if intent in REVIEW_REQUIRED_INTENTS:
        return {"action": "auto_draft_human_sends",
                "reason": f"intent '{intent}' involves money/warranty — draft is auto-generated "
                          f"but a human must approve before sending"}

    return {"action": "auto_handle",
            "reason": f"high-confidence ({confidence:.2f}) '{intent}' with strong grounding "
                      f"({top_similarity:.2f}) and no risk signals"}