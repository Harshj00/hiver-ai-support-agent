import pandas as pd
import re
import json

df = pd.read_csv('eval/golden_set_TO_LABEL.csv')

def classify_row(r):
    text = str(r['customer_text']).lower()
    supp = str(r['support_text']).lower()
    rb = r['rough_bucket']
    
    # 1. Non-English detection
    non_eng = ['hola', 'gracias', 'por favor', 'ayuda', 'bonjour', 'merci', 'obrigado', 'hilfe', 'danke', 'salut', 's\'il vous plait', 'que ', 'para ', 'con ']
    if any(kw in text for kw in non_eng) and not any(eng in text for eng in ['apple', 'iphone', 'ios', 'update', 'phone', 'help']):
        return 'other', 'yes', 'Non-English message', 'Foreign language tweet', 'Foreign language'

    # 2. Hard escalate signals
    hard_signals = ['lawyer', 'legal action', 'sue', 'attorney', 'class action', 'suicide', 'self harm', 'kill myself', 'data breach', 'hacked', 'fraud', 'unauthorized charge', 'journalist', 'press', 'reporter']
    hit_signal = next((s for s in hard_signals if s in text), None)

    # 3. Phishing / Security: explicit scam/legitimacy question, phishing, suspicious email
    if any(k in text for k in ['phishing', 'is this legit', 'suspicious email', 'fake email', 'scam email', 'fake text', 'scam text', 'is this real', 'received a text', 'got an email saying']) or ('scam' in text and any(x in text for x in ['email', 'text', 'message', 'received', 'got'])):
        return 'security_phishing', 'yes', 'Security/phishing legitimacy inquiry', 'Phishing or scam verification requested', 'Scam query'

    # 4. Account access: password, 2fa, locked out, disabled apple id, login, passcode
    if any(k in text for k in ['apple id', 'locked out', 'forgot password', 'reset password', '2fa', 'two factor', 'account disabled', 'disabled apple id', 'cant sign in', "can't sign in", 'cannot sign in', 'login problem', 'passcode', 'disabled']):
        if not any(k in text for k in ['charged', 'refund', 'money']):
            return 'account_access', 'yes' if hit_signal else 'no', hit_signal if hit_signal else '', '', 'Account access issue'

    # 5. Hardware damage: physical screen crack, dropped in water/pool, broken button, speaker physical damage
    if any(k in text for k in ['crack', 'cracked', 'shattered', 'water damage', 'pool', 'dropped in', 'dropped my', 'screen broken', 'broken screen', 'speaker broke', 'button stuck', 'hardware']):
        return 'hardware_damage', 'yes' if hit_signal else 'no', hit_signal if hit_signal else '', '', 'Physical hardware damage'

    # 6. Battery / Power: battery drain, won't charge, won't turn on, overheating
    if any(k in text for k in ['battery', 'charging', 'won\'t turn on', 'wont turn on', 'overheat', 'overheating', 'dies fast', 'drain', 'draining', 'percent']):
        if not any(b_k in text for b_k in ['charged $', 'charged me', 'refund', 'bank', 'credit card', 'statement', 'applecare']):
            return 'battery_power', 'yes' if hit_signal else 'no', hit_signal if hit_signal else '', '', 'Battery / power issue'

    # 7. Warranty / Billing: refund, charged, billing, applecare, subscription, cost, price, money back, headphones refund, digital purchase charge
    if any(k in text for k in ['refund', 'charged', 'charge', 'billing', 'applecare', 'subscription', 'invoice', 'payment', 'money back', '$', 'cost', 'buy', 'purchased', 'receipt']):
        if not any(b_kw in text for b_kw in ['battery', 'charger']):
            return 'warranty_billing', 'yes', 'Financial/billing issue involves payment or refund request', '', 'Billing/warranty question'

    # 8. Order / Shipping: order status, delivery, pre-order album, package, shipping delay
    if any(k in text for k in ['where is my order', 'delivery', 'delivered', 'shipping', 'preordered', 'pre-ordered', 'tracking', 'package', 'shipment', 'arriving', 'order']):
        return 'order_shipping', 'yes' if hit_signal else 'no', hit_signal if hit_signal else '', '', 'Order / delivery inquiry'

    # 9. Software Bug: iOS update, app crash, glitch, freeze, lag, bug, iOS version, software update
    if any(k in text for k in ['ios', 'update', 'updated', 'crash', 'crashing', 'bug', 'glitch', 'freeze', 'freezing', 'lag', 'app', 'version', 'downgrade', 'software', 'screen freeze']):
        return 'software_bug', 'yes' if hit_signal else 'no', hit_signal if hit_signal else '', '', 'Software bug / update issue'

    # 10. Feature How-To: how do i, how to, feature question
    if any(k in text for k in ['how do i', 'how to', 'is it possible', 'can i', 'how can i', 'way to']):
        return 'feature_howto', 'yes' if hit_signal else 'no', hit_signal if hit_signal else '', '', 'Feature explanation / how-to'

    # 11. Positive feedback: thanks, thank you, great, shoutout, appreciate, love
    if any(k in text for k in ['thanks', 'thank you', 'great job', 'awesome', 'shoutout', 'appreciate']):
        if not any(neg in text for neg in ['worst', 'except', 'terrible', 'issue']):
            return 'feedback_positive', 'no', '', '', 'Positive feedback'

    # 12. Negative feedback: pure complaint, venting, service criticism, "support is worst"
    if any(k in text for k in ['worst', 'terrible', 'horrible', 'disappointed', 'unacceptable', 'sucks', 'shit', 'hate', 'useless']):
        return 'feedback_negative', 'yes' if hit_signal else 'no', hit_signal if hit_signal else '', '', 'Negative feedback / customer venting'

    # Fallback to rough bucket if it's one of the 11 intents
    if rb in ['account_access', 'security_phishing', 'battery_power', 'hardware_damage', 'warranty_billing', 'software_bug', 'feature_howto', 'order_shipping', 'feedback_positive', 'feedback_negative']:
        esc = 'yes' if rb in ['security_phishing', 'warranty_billing'] or hit_signal else 'no'
        reason = 'Financial/security intent requires human review' if esc == 'yes' else ''
        return rb, esc, reason, '', f'Defaulted to rough bucket {rb}'

    return 'other', 'yes', 'Uncertain/ambiguous intent', '', 'Uncategorized message'

print("Script template ready.")
