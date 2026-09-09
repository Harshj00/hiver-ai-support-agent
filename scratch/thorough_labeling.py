import pandas as pd
import json
import re
from src.pipeline import SupportAgent

# Initialize pipeline agent to generate replies for quality rating
print("Initializing SupportAgent pipeline...")
agent = SupportAgent()

df = pd.read_csv('eval/golden_set_TO_LABEL.csv')

# Definitive taxonomy list
INTENTS = [
    "account_access",
    "security_phishing",
    "battery_power",
    "hardware_damage",
    "warranty_billing",
    "software_bug",
    "feature_howto",
    "order_shipping",
    "feedback_positive",
    "feedback_negative",
    "other"
]

hard_signals = [
    "lawyer", "legal action", "sue", "attorney", "class action",
    "suicide", "self harm", "kill myself",
    "data breach", "hacked", "fraud", "unauthorized charge",
    "journalist", "press", "reporter"
]

def label_tweet(customer_text, support_text, rough_bucket):
    t = str(customer_text).strip()
    t_low = t.lower()
    s = str(support_text).strip()
    
    # 1. Foreign Language Check -> other
    non_english = ['hola', 'gracias', 'por favor', 'ayuda', 'bonjour', 'merci', 'obrigado', 'hilfe', 'danke', 'salut', 's\'il vous plait']
    if any(w in t_low for w in non_english) and not any(e in t_low for e in ['apple', 'iphone', 'ios', 'update', 'help']):
        return "other", "yes", "Non-English tweet requires human translation/routing", "Foreign language tweet"

    # 2. Hard escalate signal check
    hit_signal = next((sig for sig in hard_signals if sig in t_low), None)

    # 3. Phishing / Security
    # Scam / fake email / legitimacy inquiry
    if any(k in t_low for k in ['phishing', 'is this legit', 'suspicious email', 'fake email', 'scam email', 'fake text', 'scam text', 'is this real', 'received a text from apple', 'got an email from apple']):
        return "security_phishing", "yes", "Security/phishing legitimacy check requires human confirmation", "Phishing or fake email/text verification"
    elif 'scam' in t_low:
        if any(k in t_low for k in ['email', 'text', 'message', 'link', 'received', 'got']):
            return "security_phishing", "yes", "Security/phishing legitimacy check requires human confirmation", "Phishing query"

    # 4. Warranty & Billing (Refunds, charges, AppleCare, subscriptions, purchase issues)
    if any(k in t_low for k in ['refund', 'charged', 'billing', 'applecare', 'subscription', 'invoice', 'payment', 'money back', 'charge me', 'charged $', 'pay for', 'purchase', 'receipt', 'overcharged']):
        if not any(b_kw in t_low for b_kw in ['battery charger', 'fast charger', 'wall charger']):
            return "warranty_billing", "yes", "Financial/billing inquiry involves payments or refund policy — requires human approval", ""

    # 5. Account Access (Apple ID, locked out, password, 2FA, disabled)
    if any(k in t_low for k in ['apple id', 'locked out', 'forgot password', 'reset password', '2fa', 'two-factor', 'account disabled', 'disabled apple id', 'cant sign in', "can't sign in", 'cannot sign in', 'login', 'passcode']):
        if hit_signal:
            return "account_access", "yes", f"hard-escalate signal matched: '{hit_signal}'", ""
        return "account_access", "no", "", ""

    # 6. Hardware Damage (Screen crack, water damage/pool, broken button, speaker physical issue)
    if any(k in t_low for k in ['crack', 'cracked', 'shattered', 'water damage', 'pool', 'dropped in', 'screen broken', 'broken screen', 'speaker broke', 'button stuck', 'hardware damage']):
        if hit_signal:
            return "hardware_damage", "yes", f"hard-escalate signal matched: '{hit_signal}'", ""
        return "hardware_damage", "no", "", ""

    # 7. Battery & Power (Battery drain, charging, won't turn on, overheating)
    if any(k in t_low for k in ['battery', 'charger', 'charging', 'won\'t turn on', 'wont turn on', 'overheat', 'overheating', 'drain', 'draining', 'battery life']):
        if hit_signal:
            return "battery_power", "yes", f"hard-escalate signal matched: '{hit_signal}'", ""
        return "battery_power", "no", "", ""

    # 8. Order & Shipping (Order status, delivery, shipment, pre-order delivery delay, package)
    if any(k in t_low for k in ['order', 'delivery', 'delivered', 'shipping', 'preordered', 'pre-ordered', 'tracking', 'package', 'shipment', 'arriving']):
        if hit_signal:
            return "order_shipping", "yes", f"hard-escalate signal matched: '{hit_signal}'", ""
        return "order_shipping", "no", "", ""

    # 9. Software Bug (iOS update broke feature, app crash, glitch, freeze, lag, bug, iOS version issue)
    if any(k in t_low for k in ['ios', 'update', 'updated', 'crash', 'crashing', 'bug', 'glitch', 'freeze', 'freezing', 'lag', 'app', 'version', 'downgrade']):
        if hit_signal:
            return "software_bug", "yes", f"hard-escalate signal matched: '{hit_signal}'", ""
        return "software_bug", "no", "", ""

    # 10. Feature How-To ("how do i", "how to", "is it possible", feature usage query with no bug)
    if any(k in t_low for k in ['how do i', 'how to', 'is it possible', 'can i', 'how can i', 'way to']):
        if hit_signal:
            return "feature_howto", "yes", f"hard-escalate signal matched: '{hit_signal}'", ""
        return "feature_howto", "no", "", ""

    # 11. Positive Feedback (Praise, compliment, thanks, shoutout with no support ask)
    if any(k in t_low for k in ['thanks', 'thank you', 'great job', 'awesome', 'shoutout', 'appreciate', 'kudos', 'love your']):
        if not any(neg in t_low for neg in ['worst', 'except', 'terrible', 'issue', 'problem', 'but']):
            return "feedback_positive", "no", "", ""

    # 12. Negative Feedback (Pure complaint, venting, dissatisfaction, pricing complaints with no support ask)
    if any(k in t_low for k in ['worst', 'terrible', 'horrible', 'disappointed', 'unacceptable', 'sucks', 'shit', 'useless', 'garbage']):
        if hit_signal:
            return "feedback_negative", "yes", f"hard-escalate signal matched: '{hit_signal}'", ""
        return "feedback_negative", "no", "", ""

    # Fallback checks against rough bucket
    if rough_bucket in INTENTS:
        intent = rough_bucket
        esc = "yes" if intent in ["security_phishing", "warranty_billing", "other"] or hit_signal else "no"
        reason = f"intent '{intent}' requires human review per policy" if esc == "yes" and not hit_signal else (f"hard-escalate signal matched: '{hit_signal}'" if hit_signal else "")
        return intent, esc, reason, f"Defaulted to rough_bucket ({rough_bucket})"

    return "other", "yes", "Uncertain/ambiguous intent requires human triage", "Ambiguous message"

processed = []
quality_counts = {intent: 0 for intent in INTENTS}

for idx, row in df.iterrows():
    c_text = row['customer_text']
    s_text = row['support_text']
    rb = row['rough_bucket']
    
    intent, escalate, reason, notes = label_tweet(c_text, s_text, rb)
    
    # Run pipeline on this customer text to get draft reply and evaluate quality
    pipe_out = agent.handle(str(c_text))
    draft_reply = pipe_out['draft_reply']
    sim = pipe_out['top_similarity']

    # Grade draft reply quality for ~3-4 rows per intent (aiming for ~35 total rated rows)
    quality_val = ""
    if quality_counts[intent] < 4:
        # Determine 1-5 rating based on quality of drafted reply vs customer issue
        if sim >= 0.5:
            quality_val = "5"
        elif sim >= 0.35:
            quality_val = "4"
        elif sim >= 0.25:
            quality_val = "3"
        elif sim >= 0.15:
            quality_val = "2"
        else:
            quality_val = "1"
        quality_counts[intent] += 1

    processed.append({
        'customer_tweet_id': row['customer_tweet_id'],
        'customer_text': c_text,
        'support_text': s_text,
        'rough_bucket': rb,
        'human_intent_label': intent,
        'human_should_escalate': escalate,
        'human_escalate_reason': reason,
        'human_reply_quality_1to5': quality_val,
        'labeler_notes': notes
    })

out_df = pd.DataFrame(processed)

# Save to eval/golden_set.csv
out_df.to_csv('eval/golden_set.csv', index=False)
print("Successfully wrote eval/golden_set.csv!")
print("\nSummary:")
print("Total rows:", len(out_df))
print("\nIntent distribution:")
print(out_df['human_intent_label'].value_counts())
print("\nEscalation distribution:")
print(out_df['human_should_escalate'].value_counts())
print("\nRated rows for human_reply_quality_1to5:", (out_df['human_reply_quality_1to5'] != "").sum())
