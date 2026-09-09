"""
Thin wrapper around the Anthropic API.

Decision (see DECISION_LOG.md #9): if ANTHROPIC_API_KEY is not set, the
pipeline does NOT crash — it falls back to a deterministic, clearly-labelled
stub so `README` reproduction still finishes in minutes for a first smoke
test. Any real evaluation number MUST be produced with a real key
(USING_LLM=True in the output), or it doesn't count.
"""
import os

_client = None
USING_LLM = False

try:
    import anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        _client = anthropic.Anthropic()
        USING_LLM = True
except ImportError:
    pass


def complete(system: str, user: str, model: str, max_tokens: int = 500) -> str:
    if _client is not None:
        resp = _client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
    return _stub_complete(system, user)


def _stub_complete(system: str, user: str) -> str:
    """
    Deterministic offline fallback — NOT a substitute for the real model.
    Returns an obviously-a-stub string so it can never be mistaken for a
    real judged/graded output.
    """
    return "[OFFLINE STUB — set ANTHROPIC_API_KEY for a real response]"
