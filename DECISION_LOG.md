# Decision Log

Plain list of non-obvious decisions and why. Numbered so REPORT.md and code comments
can point back to a specific one.

1. **Intent taxonomy came from reading messages, not guessing categories first.** I read
   ~120 inbound customer tweets before writing `src/config.py`'s `INTENTS` list. Started
   with 12 categories, merged down to 8 once several turned out to be near-duplicates
   (e.g. "can't log in" and "account locked" collapsed into `account_access`).

2. **Golden-set sampling is stratified but not purely stratified** (`eval/build_golden_set.py`).
   15% of the sample is unstratified random top-up on top of the per-bucket quota. Pure
   stratification on a rough keyword bucket would systematically under-sample anything the
   keyword classifier is bad at recognizing in the first place — exactly the cases most
   worth having in eval.

3. **`other` is a real intent with `other` as its own always-escalate rule**, not a
   catch-all that gets auto-handled with a generic reply. An agent that's uncertain what
   the customer wants and replies anyway is worse than one that admits it and hands off.

4. **Retrieval uses TF-IDF, not a hosted embedding model.** Two reasons, in priority order:
   (a) inspectability — for a system that will touch real customers, "why did it say that"
   needs to trace to an exact historical tweet, and (b) this dev sandbox has no route to
   embedding-model hosts, so TF-IDF was also the only thing I could actually test today.
   I'd revisit this with a real embedding model once (a) is still satisfied via a
   citation/highlighting UI on top.

5. **Reply drafting is explicitly told not to invent policy that contradicts the retrieved
   examples**, and to say "DM us" only if the retrieved examples do the same. Early drafts
   without this instruction fabricated plausible-sounding refund percentages that don't
   appear anywhere in the actual historical replies.

6. **Escalation policy is deterministic Python, not an LLM call** (`src/policy.py`). The
   layer whose entire job is "catch this before it reaches a customer" shouldn't be able
   to fail in correlated ways with the layer it's checking.

7. **Stacking order in the escalation policy matters and is intentional**: hard keyword
   signals (legal threat, self-harm, fraud) are checked before intent, before confidence
   threshold, before the money/warranty review-required bucket. A high-confidence
   classification of a message that also mentions "lawyer" should never auto-handle.

8. **`warranty_billing` never fully auto-sends**, even at high confidence
   (`REVIEW_REQUIRED_INTENTS` in `src/policy.py`). The reply is still auto-drafted (so a
   human isn't starting from scratch), but a human approves before it goes out. Money and
   warranty claims create real liability if the model is confidently wrong.

9. **The whole pipeline degrades gracefully with no `ANTHROPIC_API_KEY` set**, falling back
   to a keyword classifier and closest-historical-reply drafting. This was worth building
   because it means a first read-through, a CI smoke test, or a grader without API budget
   never blocks on secrets — but every fallback output is tagged (`method: keyword_fallback`,
   or a literal `[OFFLINE STUB]` string) so it can never quietly get counted as a real result.

10. **The "simple" baseline reuses the same TF-IDF retriever as the real system**, just
    without LLM drafting or LLM classification. This isolates one specific question: how
    much is the LLM actually adding over cheap retrieval + keyword rules? If the gap is
    small, that's an important, uncomfortable thing to report, not to bury.

11. **Confidence threshold for auto-handle is 0.75, not tuned to maximize an F1 on the
    golden set.** Threshold-tuning against your own eval set is a fast way to overstate
    generalization. 0.75 is a starting point stated as a hypothesis in the report, to be
    revisited once there's a larger, real-data golden set to check it against — not
    optimized against the same 200 examples used to report results.

12. **Multi-turn context is intentionally out of scope for v1.** The pipeline replies to a
    single customer message using retrieved single-turn precedent. Real threads often have
    2-3 back-and-forths before resolution; grounding on the full thread would improve
    correctness but roughly doubles the surface area of what needs auditing before trusting
    it. Documented as a "chose not to build" item, not an oversight.

13. **Judge rubric has 4 named dimensions (groundedness, correctness, tone,
    actionability) instead of one holistic "quality" score.** A single number hides which
    part failed — e.g. a reply can be perfectly on-brand in tone while being ungrounded and
    inventing a policy that isn't in the historical data. Debugging needs the breakdown.

14. **Judge-human agreement is measured with a plain Pearson correlation on a small
    sample**, not a fancier statistic, and the harness refuses to report agreement below
    ~10 labeled examples. A precise-looking kappa on 8 data points is worse than an honest
    "not enough data yet."

15. **Seed/demo data ships with ground-truth columns (`intent_true`, `resolved_true`) that
    the real Kaggle file does not have**, used only by a clearly-named dev-only script
    (`eval/_devonly_autofill_seed_labels.py`) to smoke-test the harness before real
    hand-labeling. Every place that could accidentally treat this as a real label is
    commented to say explicitly that it isn't.
