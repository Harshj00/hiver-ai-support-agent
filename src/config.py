"""
Single source of truth for brand + taxonomy.
Swap BRAND to target a different company from the Kaggle dataset —
nothing else in the pipeline needs to change.
"""

BRAND = "AppleSupport"          # Twitter handle of the support account in twcs.csv
BRAND_AUTHOR_FIELD = "author_id"

RAW_CSV_PATH = "data/twcs.csv"          # real Kaggle file goes here (see README)
SEED_CSV_PATH = "data/seed_conversations.csv"  # synthetic stand-in shipped with repo
GOLDEN_SET_PATH = "eval/golden_set.csv"
GOLDEN_SAMPLE_PATH = "eval/golden_set_TO_LABEL.csv"

# Intents were NOT invented in the abstract — they came from open-coding ~120
# customer-inbound tweets to AppleSupport by hand (see DECISION_LOG.md, item 1),
# then REVISED after reviewing a real stratified sample from the actual Kaggle
# data (DECISION_LOG.md, item 18) — two categories (security_phishing,
# feedback_negative) were added because the original 8 were structurally
# forcing real messages into the wrong bucket, not just mislabeling edge cases.
INTENTS = [
    "account_access",       # forgot password / 2FA locked out / can't sign in to Apple ID
    "security_phishing",    # suspected scam email/text, "is this legit", unauthorized access concern
    "battery_power",        # battery drains fast / won't charge / won't turn on
    "hardware_damage",      # cracked screen, water damage, physical repair
    "warranty_billing",     # AppleCare status, unexpected charge, refund request
    "software_bug",         # app crash, iOS bug, update broke something
    "feature_howto",        # "how do I..." / feature explanation, no bug or account issue
    "order_shipping",       # where's my order / delivery delay / wrong item
    "feedback_positive",    # genuine compliment / thanks, no ask
    "feedback_negative",    # complaint, venting, service-quality criticism — not a bug report, not praise
    "other",                # doesn't fit cleanly — always route to human by default
]

# Messages containing these signals should never be auto-resolved regardless
# of intent confidence — see DECISION_LOG.md, item 6.
HARD_ESCALATE_SIGNALS = [
    "lawyer", "legal action", "sue", "attorney", "class action",
    "suicide", "self harm", "kill myself",
    "data breach", "hacked", "fraud", "unauthorized charge",
    "journalist", "press", "reporter",
]

AUTO_HANDLE_CONFIDENCE_THRESHOLD = 0.75
MIN_GROUNDING_SIMILARITY = 0.30  # provisional — see DECISION_LOG.md #17
MODEL = "claude-sonnet-4-5"
JUDGE_MODEL = "claude-sonnet-4-5"