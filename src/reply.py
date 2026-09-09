from src.config import MODEL
from src.llm_client import complete, USING_LLM

SYSTEM = (
    "You are drafting a public Twitter reply for a brand's customer support account. "
    "Match the brand's historical tone and structure shown in the examples below — do not "
    "invent policy that contradicts them. Keep it under 280 characters, no hashtags, "
    "ask the customer to DM only if the examples do. If none of the examples resolve this "
    "specific case, say so plainly and suggest DMing rather than guessing at a fix."
)


def draft_reply(customer_text: str, retrieved) -> dict:
    examples = "\n\n".join(
        f"Similar past customer message: \"{r.customer_text}\"\n"
        f"Brand's actual reply: \"{r.support_text}\""
        for r in retrieved.itertuples()
    )
    prompt = (
        f"Historical grounding examples (most similar first):\n{examples}\n\n"
        f"New customer message to reply to:\n\"{customer_text}\"\n\n"
        f"Draft the reply now (text only, no preamble)."
    )
    if USING_LLM:
        text = complete(SYSTEM, prompt, model=MODEL, max_tokens=200)
    else:
        # offline fallback: closest historical reply, lightly marked as a template
        top = retrieved.iloc[0]
        text = top["support_text"] if top["similarity"] > 0.15 else \
            "Thanks for reaching out — a team member will follow up shortly. (offline stub, no LLM key set)"
    return {
        "reply": text,
        "grounded_on": retrieved["customer_text"].tolist(),
        "top_similarity": float(retrieved["similarity"].iloc[0]) if len(retrieved) else 0.0,
    }
