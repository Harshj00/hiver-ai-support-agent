# Golden Set Labeling Guide

Read this once before you start labeling `eval/golden_set_TO_LABEL.csv`. Budget 2-3 min/row.
Consistency across 200 rows matters more than getting each individual row "perfect."

## human_intent_label
Pick exactly one from `src/config.py`'s `INTENTS` list. Rules for the ambiguous cases:

- **Tone doesn't override intent.** A furious, sarcastic, or sweary message that contains a
  specific fixable technical complaint still gets the technical intent
  (`software_bug`, `battery_power`, `hardware_damage`, etc.), not `feedback_negative`.
  Example: *"Thats terrible. I can't upload photos on Instagram anymore"* → `software_bug`,
  not `feedback_negative` — there's a concrete broken thing to fix.
- **`feedback_negative` is for when there's nothing specific to fix.** Pure venting, "your
  support is the worst," pricing complaints with no support ask. If you can't write a
  one-sentence troubleshooting next step, it's probably `feedback_negative` or `other`.
- **`security_phishing` requires an actual scam/legitimacy question**, not just the word
  "scam" used as an insult ("y'all some scammers" about slow battery life is `battery_power`,
  not `security_phishing` — check what the customer is actually asking).
- **`feature_howto` is for "how do I / is X possible" questions with no bug or account
  problem.** If the "how" is really "how do I fix this broken thing," it's the technical
  intent, not `feature_howto`.
- **Non-English tweets** → `other`, unless you can read the language and it clearly maps to
  a specific intent — don't guess from cognates alone.
- If genuinely torn between two intents, pick the one a human agent would file it under for
  routing purposes, and note the ambiguity in `labeler_notes`. Don't invent a 12th category.

## human_should_escalate (yes/no) + human_escalate_reason
Ask: **would you be comfortable if this exact reply went out to the customer with no human
review?** If there's real ambiguity, financial stakes, safety language, or the retrieved
precedent doesn't actually match the specific problem — escalate. Write the reason in your
own words; it doesn't need to match the system's stated categories.

## human_reply_quality_1to5 (only needed for ~30 rows, for judge-agreement check)
Rate the reply the pipeline actually drafted for this row (run the pipeline first, see
README step 2) on: does it address the real issue, is it grounded in something plausible
for this brand, would a human need to substantially rewrite it before sending.
- 5 = ready to send as-is
- 3 = right direction, needs editing
- 1 = wrong or unsafe

## labeler_notes
Use freely for anything that felt ambiguous, mislabeled by the rough_bucket keyword guesser,
or worth mentioning in the failure analysis section of the report. This column is where your
"top 5 failure modes" examples should come from — don't discard your own uncertainty.