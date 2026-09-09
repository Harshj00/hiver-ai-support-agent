# Decision Log

A list of the non-evident decisions along with the reasons for them. Included here for use in REPORT.md and in the code comments.
one can refer to a particular instance.

1. The intent taxonomy was derived from reading the messages, not by first guessing the categories. I read
   Before writing the `INTENTS` list in `src/config.py` there were about 120 inbound customer tweets. I started.
   with 12 categories, merged down to 8 once several turned out to be near-duplicates
   For example, 'can't log in' and 'account locked' are combined under `account_access`.

2. **Golden-set sampling is stratified, but not fully stratified** (`eval/build_golden_set.py`).
   15 percent of the sample consists of unstratified random top-ups in addition to the per-bucket quota. Certainly.
   stratification on a rough keyword bucket would systematically under-sample anything the
   keyword classifier is bad at recognizing in the first place - exactly the cases most
   worth having in eval.

Other is a real intent having other as its own rule which always escalates.
   A catch-all item which is automatically dealt with by a general reply. The case of an agent who doesn't know what
   It is better for the customer to admit it and transfer the issue than it is for them to say they want it and then reply.

4. The retrieval process uses TF-IDF and not a hosted embedding model. This is for the following two reasons, ranked in order of priority:
   (a) inspectability - for a system that will touch real customers, "why did it say that"
   needs to trace to an exact historical tweet, and (b) this dev sandbox has no route to
   Since the embedding model hosts were the only option, I was also able to test only the TF-IDF.
   I'd revisit this with a real embedding model once (a) is still satisfied via a
   citation/highlighting UI on top.

5. It is specifically stated that reply drafting should not invent policy which contradicts the retrieved
   Examples, and in each case it is only necessary to say "DM us" if the examples that have been retrieved also do so. Early drafts
   without this instruction fabricated plausible-sounding refund percentages that don't
   Turn up in the real historical replies.

The escalation policy is deterministic Python, not a call to an LLM (in src/policy.py). The
   layer whose entire job is "catch this before it reaches a customer" shouldn't be able
   to fail in a way that is correlated with the layer it is checking.

7. The order in which the escalation policy steps are stacked is important and has been deliberately chosen: hard keyword
   signals (legal threat, self-harm, fraud) are checked before intent, before confidence
   At the threshold, prior to the money/warranty review-required bucket. A high-confidence
   The message should never be automatically handled if it refers to a 'lawyer'.

The warranty_billing feature never fully auto-sends, even when the confidence level is high.
   The review is still automatically drafted (so a
   a human doesn't have to start from scratch, but approval from a human is required before it is released. Money and
   There is actual liability involved when warranty claims are made on the basis of the model being definitely incorrect.

9. The entire pipeline works smoothly even if the ANTHROPIC_API_KEY is not set, switching to a fallback mechanism
   to a keyword classifier and one that drafts the most closely matching historical reply. This was worth the effort of building
   because it means a first read-through, a CI smoke test, or a grader without API budget
   never blocks on secrets - but every fallback output is tagged (`method: keyword_fallback`,
   or a literal `[OFFLINE STUB]` string, so that it can never be counted as a real result without being noticed.

10. **The simple baseline makes use of the same TF-IDF retriever as the actual system**, just
    Without the use of LLMs for drafting or for classification, we can focus on one particular question: how
    How much does the LLM actually add on top of using cheap retrieval combined with keyword rules? If the gap is
    It's small and that's an important and uncomfortable thing to have to report, not to bury.

11. The confidence threshold for auto-handle is 0.75, not having been tuned to maximise an F1 score.
    The golden set. Adjusting the threshold using your own evaluation set is a quick way of overestimating
    0.75 is given in the report as a hypothesis and stated as a starting point.
    revisited once there's a larger, real-data golden set to check it against - not
    Optimized using the same 200 examples that were used to report the results.

For version v1, multi-turn context is deliberately not included. The pipeline responds to a
    A single-customer message is created using the retrieved single-turn precedent. Generally, real threads have
    2-3 back-and-forths before resolution; grounding on the full thread would improve
    correctness but roughly doubles the surface area of what needs auditing before trusting
    It is listed as an item that had chosen not to be built, not as a mistake.

13. The judge's rubric includes four named dimensions: groundedness, correctness, tone, and
    Instead, they use separate actionability scores rather than a single overall "quality" score since a single number masks which
    A component failed - for example, a reply can be exactly on-brand in terms of tone and yet remain ungrounded and
    To create a policy that isn't found in the historical data, debugging requires a breakdown.

14. The degree of agreement between the judge and the human is determined using a simple Pearson correlation on a small
    sample**, not a fancier statistic, and the harness refuses to report agreement below
    There are about 10 labeled examples; a kappa value that appears precise based on 8 data points is in fact worse than an honest one.
    "not enough data yet."

15. The seed/demo data includes columns containing the ground truth (intent_true, resolved_true) that
    the real Kaggle file does not have**, used only by a clearly-named dev-only script
    To smoke-test the harness before real
    By manually labeling. Every situation in which this might accidentally be taken as a real label is
    said outright that it isn't.
