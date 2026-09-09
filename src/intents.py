import json
import re
from src.config import INTENTS, MODEL
from src.llm_client import complete, USING_LLM

INTENT_DEFS = {
    "account_access": "Locked out, forgot password, 2FA / sign-in problems, disabled Apple ID.",
    "security_phishing": "Suspects a scam email/text/call, asks 'is this legit', unauthorized access worry.",
    "battery_power": "Battery drains fast, won't charge, won't turn on, overheating.",
    "hardware_damage": "Physical damage: cracked screen, water damage, broken buttons.",
    "warranty_billing": "AppleCare coverage questions, unexpected charges, refund requests.",
    "software_bug": "App crashes, OS bugs introduced by an update, broken features after update.",
    "feature_howto": "Asking how to do something or how a feature works — not a bug, not an account problem.",
    "order_shipping": "Order status, delivery delay, wrong item, address changes.",
    "feedback_positive": "Genuine compliment or thanks, no support ask attached.",
    "feedback_negative": "Complaint, venting, or criticism of service/product/pricing — not a specific bug report.",
    "other": "Doesn't fit cleanly into the above, or is off-topic.",
}

SYSTEM = (
    "You are an intent classifier for a customer support Twitter account. "
    "Classify the customer's message into exactly one intent from the provided list. "
    "Respond ONLY with compact JSON: {\"intent\": \"<one of the labels>\", "
    "\"confidence\": <0.0-1.0>, \"rationale\": \"<one short sentence>\"}. "
    "No prose, no markdown fences."
)


def _prompt(text: str) -> str:
    defs = "\n".join(f"- {k}: {v}" for k, v in INTENT_DEFS.items())
    return f"Intent definitions:\n{defs}\n\nCustomer message:\n\"\"\"{text}\"\"\""


def classify(text: str) -> dict:
    if USING_LLM:
        raw = complete(SYSTEM, _prompt(text), model=MODEL, max_tokens=150)
        try:
            parsed = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
            if parsed.get("intent") not in INTENTS:
                parsed["intent"] = "other"
            return {
                "intent": parsed["intent"],
                "confidence": float(parsed.get("confidence", 0.5)),
                "rationale": parsed.get("rationale", ""),
                "method": "llm",
            }
        except Exception:
            return {"intent": "other", "confidence": 0.0,
                     "rationale": "LLM output unparseable — routed to other as a safe default.",
                     "method": "llm_parse_failed"}
    return _fallback_classify(text)


# --- offline fallback: also doubles as the "simple baseline" in eval ---
_KEYWORDS = {
    "account_access": ["password", "locked out", "sign in", "2fa", "apple id", "account disabled"],
    "security_phishing": ["phishing", "scam", "fraud", "suspicious email", "hacked", "verify your account"],
    "battery_power": ["battery", "charge", "charging", "won't turn on", "overheat"],
    "hardware_damage": ["crack", "screen", "broke", "dropped", "water damage", "button stuck"],
    "warranty_billing": ["charged", "refund", "applecare", "warranty", "billing", "price"],
    "software_bug": ["crash", "bug", "update", "ios", "lag", "glitch", "freeze", "restart"],
    "feature_howto": ["how do i", "how to", "is there a way to"],
    "order_shipping": ["order", "shipping", "delivery", "delivered", "package"],
    "feedback_positive": ["thanks", "thank you", "great", "shoutout", "appreciate"],
    "feedback_negative": ["worst", "terrible", "ridiculous", "disappointed", "unacceptable"],
}


def _fallback_classify(text: str) -> dict:
    t = text.lower()
    scores = {intent: sum(1 for kw in kws if kw in t) for intent, kws in _KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return {"intent": "other", "confidence": 0.3, "rationale": "no keyword matched",
                "method": "keyword_fallback"}
    conf = min(0.5 + 0.15 * scores[best], 0.9)
    return {"intent": best, "confidence": conf, "rationale": f"matched keyword(s) for {best}",
             "method": "keyword_fallback"}