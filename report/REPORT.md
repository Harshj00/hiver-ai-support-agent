# Report — AppleSupport AI Support Agent

> **Numbers in this document were produced against the real Kaggle customer support dataset (`data/twcs.csv`) and the hand-labeled Golden Set (`eval/golden_set.csv`, n=195)**.

## 1. Problem framing

**What "good" means for this brand.** AppleSupport's public Twitter replies are short,
consistent in structure (acknowledge → concrete next step → invite a DM for anything
account-specific), and almost never resolve a ticket fully in public — the public tweet's
job is triage, not resolution. So "good" for this agent means: (a) correctly bucket the
customer's problem, (b) produce a reply that matches that triage pattern using real
precedent, not invented policy, and (c) be conservative about what it auto-sends — a wrong
public reply is worse than a slightly-slower human reply.

**What I chose not to build:**
- Multi-turn thread modeling (see Decision Log #12) — v1 replies to one message at a time.
- A fine-tuned classifier — the intent set will likely shift once real data is reviewed
  properly, and prompted classification is cheaper to iterate on than retraining.
- Auto-sending anything involving money (Decision Log #8) — draft-and-review only.
- Personalization beyond what's in the tweet itself (no CRM/order-history lookup) — out of
  scope for a Twitter-only agent working from this dataset alone.

## 2. Results vs. baselines

| System | Intent accuracy | Intent macro-F1 | Escalation precision | Escalation recall |
|---|---|---|---|---|
| Trivial (majority intent, always escalate) | 0.11 | 0.02 | 0.32 | 1.00 |
| Simple (keyword intent + verbatim retrieval, keyword escalation) | 0.76 | 0.77 | 0.83 | 0.38 |
| **Pipeline (LLM intent + grounded LLM reply + policy)** | 0.76* | 0.77* | 0.35 | **0.98** |

\* *When `ANTHROPIC_API_KEY` is unconfigured, the pipeline's intent classification path falls back to the deterministic keyword classifier (`src.intents._fallback_classify`) as documented in Decision Log #9. Consequently, intent classification accuracy is identical between Simple and Pipeline by construction on offline fallback runs.*

The key structural differentiator between the **Simple** baseline and the **Pipeline** is the **escalation decision safety**:
- The **Simple baseline** relies solely on crude keyword detection for escalation, achieving a dangerously low **escalation recall of 0.38** (missing 62% of messages that require human intervention, including complex disputes and ambiguous issues).
- The **Pipeline policy** enforces deterministic safeguards (`security_phishing`, `warranty_billing`, `other`, low-confidence thresholds, and hard risk signals), achieving an **escalation recall of 0.98** — ensuring near-total coverage for high-risk customer interactions before public auto-reply.

## 3. Failure analysis — top 5 modes, with real examples from the golden set

1. **Politeness phrases in follow-up tweets override technical intent (`software_bug` → `feedback_positive`).**
   - *Example:* `"Appreciate your reply but I've managed to troubleshoot it myself. Lost almost all my data though"` was classified as `feedback_positive` because of the lead word "Appreciate". The customer actually suffered major data loss during an iOS update troubleshooting process.
   - *Hypothesis:* Politeness detectors must evaluate sentence context or require dedicated negative sentiment / error indicator overrides before routing to `feedback_positive`.

2. **Overloaded domain terms causing keyword misclassifications (`screen` keyword collision).**
   - *Example:* `"I turned off screen recording but everyone on my Snapchat keeps getting notified"` was classified as `hardware_damage` because "screen" triggered the physical screen damage keyword list.
   - *Hypothesis:* Word-boundary matching and contextual disambiguation (e.g. distinguishing "screen recording" or "screen time" from "cracked screen") are necessary for exact feature classification.

3. **Sub-string collisions on financial vs. power terms ("charged $9.99" vs. battery "charge").**
   - *Example:* `"Just got charged $9.99 for storage I didn't order"` matching battery "charge" in naive substring matchers.
   - *Hypothesis:* Strict regex token boundaries with stop-word exclusions for financial numbers (`$`, `billing`, `card`) must take precedence over hardware power keywords.

4. **In-app authentication prompts misidentified as Apple ID account access.**
   - *Example:* `"how do I know what app is requesting my password? This is a bit scary..."` was classified as `account_access` rather than `software_bug` or security anomaly because of the word "password".
   - *Hypothesis:* Expand intent boundaries for `software_bug` / `security_phishing` to cover unexpected OS dialogs and third-party app credential prompts.

5. **Digital pre-order delivery delays colliding with physical order shipping (`order_shipping` vs `warranty_billing`).**
   - *Example:* `"I preordered the reputation album but Call It What You Want isn't in my library... WHY?!"` was bucketed under `order_shipping` due to "preordered", whereas digital content issues are handled via iTunes / App Store billing workflows.
   - *Hypothesis:* Disambiguate digital iTunes / App Store content fulfillment from physical product shipping inquiries in the retrieval and classification schema.

## 4. What's misleading about my headline numbers (mandatory section)

1. **Escalation recall of 0.98 is driven by strict policy rules, not LLM intelligence.** Because `warranty_billing`, `security_phishing`, and `other` always escalate, policy rules catch almost all risky cases deterministically. This yields high safety (recall=0.98) at the expense of escalation precision (0.35), meaning a substantial portion of customer messages are sent to human queues.

2. **Intent accuracy parity between Simple and Pipeline (0.76) reflects the offline fallback path.** When running without an active API key, both systems rely on keyword classification. Evaluating with live API calls will demonstrate the true macro-F1 delta of LLM prompt-based classification over keyword matching.

3. **The Golden Set (n=195) provides a stratified sample, but real TWCS volume exhibits long-tail variance.** While 195 hand-labeled rows give reliable directionality for core intents, rare edge cases and multi-turn conversational nuances require expanding the golden set as edge cases emerge in production.

4. **Human Quality Rating (`human_reply_quality_1to5`) sample (n=44):** Rated replies indicate high precedent grounding for standard intent buckets (e.g. software bugs and battery queries scoring 4-5), but lower scores (1-2) on ungrounded edge cases where retrieved historical tweets did not match the specific sub-issue.

## 5. What I'd do next with one more week

- Activate live Anthropic API evaluation across the 195 golden set rows to measure the exact LLM intent classification accuracy and LLM-as-judge agreement score (Pearson correlation $r$).
- Implement context-aware sentiment filtering to prevent politeness terms ("thanks", "appreciate") from misclassifying technical bug follow-ups as positive feedback.
- Add regex word-boundary matching to resolve substring collisions between battery charging and billing charges.
- Separate `warranty_billing` into `billing_inquiry` (auto-handleable informational queries) vs `billing_dispute` (refund/unauthorized charge requests requiring mandatory human escalation) to improve escalation precision without sacrificing safety.
- Introduce minimum cosine/TF-IDF similarity thresholds in `src/retrieval.py` to trigger fallback escalation whenever historical precedent similarity drops below 0.30.
