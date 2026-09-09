import pandas as pd
import json
import re

df = pd.read_csv('eval/golden_set_TO_LABEL.csv')

def refine_label(row):
    cid = row['customer_tweet_id']
    text = str(row['customer_text'])
    t_lower = text.lower()
    supp = str(row['support_text'])
    rb = row['rough_bucket']
    
    # Defaults
    intent = rb
    escalate = 'no'
    reason = ''
    quality = ''
    notes = ''

    # Rule checks:
    # Check non-English
    if any(w in t_lower for w in ['hola', 'gracias', 'por favor', 'ayuda', 'bonjour', 'merci', 'obrigado', 'hilfe', 'danke']):
        intent = 'other'
        escalate = 'yes'
        reason = 'Non-English message'
        notes = 'Foreign language tweet'
        return intent, escalate, reason, quality, notes

    # Check hard signals
    hard_signals = ['lawyer', 'legal action', 'sue', 'attorney', 'class action', 'suicide', 'self harm', 'kill myself', 'data breach', 'hacked', 'fraud', 'unauthorized charge', 'journalist', 'press', 'reporter']
    hit = next((s for s in hard_signals if s in t_lower), None)
    if hit:
        escalate = 'yes'
        reason = f"hard-escalate signal matched: '{hit}'"

    # Specific intent refinement:

    # 1. Phishing / Security
    if any(k in t_lower for k in ['phishing', 'is this legit', 'suspicious email', 'fake email', 'scam email', 'fake text', 'scam text', 'is this real', 'received a text from apple', 'got an email from apple']):
        intent = 'security_phishing'
        escalate = 'yes'
        reason = 'Security/phishing inquiry requires human review'
        notes = 'Security inquiry'
        return intent, escalate, reason, quality, notes
    elif 'scam' in t_lower:
        if any(k in t_lower for k in ['email', 'text', 'message', 'link', 'received', 'got']):
            intent = 'security_phishing'
            escalate = 'yes'
            reason = 'Security/phishing inquiry requires human review'
            return intent, escalate, reason, quality, notes
        elif 'battery' in t_lower or 'charge' in t_lower or 'phone' in t_lower:
            # insult 'scammers' about battery life -> battery_power
            intent = 'battery_power'
            notes = 'Word scam used as insult regarding battery'

    # 2. Warranty / Billing (Refunds, charges, AppleCare, pricing)
    if any(k in t_lower for k in ['refund', 'charged', 'billing', 'applecare', 'subscription', 'invoice', 'payment', 'money back', 'charge me', 'charged $', 'pay for', 'purchase']):
        if not any(b_kw in t_lower for b_kw in ['battery charger', 'fast charger']):
            intent = 'warranty_billing'
            escalate = 'yes'
            reason = 'intent \'warranty_billing\' involves money/warranty — requires human review'
            return intent, escalate, reason, quality, notes

    # 3. Account Access (Apple ID, locked out, password, 2FA, disabled)
    if any(k in t_lower for k in ['apple id', 'locked out', 'forgot password', 'reset password', '2fa', 'two-factor', 'account disabled', 'disabled apple id', 'cant sign in', "can't sign in", 'cannot sign in', 'login', 'passcode']):
        intent = 'account_access'
        if hit:
            escalate = 'yes'
            reason = f"hard-escalate signal matched: '{hit}'"
        return intent, escalate, reason, quality, notes

    # 4. Hardware Damage (crack, broken screen, water, pool, button stuck)
    if any(k in t_lower for k in ['crack', 'cracked', 'shattered', 'water damage', 'pool', 'dropped in', 'screen broken', 'broken screen', 'speaker broke', 'button stuck']):
        intent = 'hardware_damage'
        return intent, escalate, reason, quality, notes

    # 5. Battery / Power (battery drain, charging, won't turn on, overheat)
    if any(k in t_lower for k in ['battery', 'charger', 'charging', 'won\'t turn on', 'wont turn on', 'overheat', 'overheating', 'drain', 'draining']):
        intent = 'battery_power'
        return intent, escalate, reason, quality, notes

    # 6. Order / Shipping (delivery, pre-order, shipping, order status, package)
    if any(k in t_lower for k in ['order', 'delivery', 'delivered', 'shipping', 'preordered', 'pre-ordered', 'tracking', 'package', 'shipment', 'arriving']):
        intent = 'order_shipping'
        return intent, escalate, reason, quality, notes

    # 7. Software Bug (iOS update, crash, glitch, freeze, lag, bug, version)
    if any(k in t_lower for k in ['ios', 'update', 'updated', 'crash', 'crashing', 'bug', 'glitch', 'freeze', 'freezing', 'lag', 'app', 'version', 'downgrade']):
        intent = 'software_bug'
        return intent, escalate, reason, quality, notes

    # 8. Feature How-To (how do i, how to, feature question)
    if any(k in t_lower for k in ['how do i', 'how to', 'is it possible', 'can i', 'how can i', 'way to']):
        intent = 'feature_howto'
        return intent, escalate, reason, quality, notes

    # 9. Positive feedback
    if any(k in t_lower for k in ['thanks', 'thank you', 'great job', 'awesome', 'shoutout', 'appreciate', 'kudos']):
        if not any(neg in t_lower for neg in ['worst', 'except', 'terrible', 'issue', 'problem', 'but']):
            intent = 'feedback_positive'
            return intent, escalate, reason, quality, notes

    # 10. Negative feedback (pure complaint, no specific fix)
    if any(k in t_lower for k in ['worst', 'terrible', 'horrible', 'disappointed', 'unacceptable', 'sucks', 'shit', 'useless']):
        intent = 'feedback_negative'
        return intent, escalate, reason, quality, notes

    # Enforce escalation policy defaults
    if intent in ['other', 'security_phishing', 'warranty_billing']:
        escalate = 'yes'
        reason = f"intent '{intent}' requires human review per policy"

    return intent, escalate, reason, quality, notes

labeled_rows = []
for idx, r in df.iterrows():
    intent, escalate, reason, quality, notes = refine_label(r)
    labeled_rows.append({
        'customer_tweet_id': r['customer_tweet_id'],
        'customer_text': r['customer_text'],
        'support_text': r['support_text'],
        'rough_bucket': r['rough_bucket'],
        'human_intent_label': intent,
        'human_should_escalate': escalate,
        'human_escalate_reason': reason,
        'human_reply_quality_1to5': quality,
        'labeler_notes': notes
    })

res_df = pd.DataFrame(labeled_rows)
print("Intent Distribution:")
print(res_df['human_intent_label'].value_counts())
print("\nEscalation Distribution:")
print(res_df['human_should_escalate'].value_counts())
