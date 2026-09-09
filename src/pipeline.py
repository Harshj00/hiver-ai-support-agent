import pandas as pd
from src.ingest import load_raw, build_pairs
from src.retrieval import ReplyRetriever
from src.intents import classify
from src.reply import draft_reply
from src.policy import decide


class SupportAgent:
    def __init__(self):
        df, self.is_real_data = load_raw()
        self.pairs = build_pairs(df)
        if len(self.pairs) == 0:
            raise RuntimeError("No resolved conversation pairs found for this brand.")
        self.retriever = ReplyRetriever(self.pairs)

    def handle(self, customer_text: str, k: int = 3) -> dict:
        intent_result = classify(customer_text)
        retrieved = self.retriever.top_k(customer_text, k=k)
        reply_result = draft_reply(customer_text, retrieved)
        decision = decide(intent_result["intent"], intent_result["confidence"],
                          customer_text, reply_result["top_similarity"])
        return {
            "customer_text": customer_text,
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "intent_method": intent_result["method"],
            "draft_reply": reply_result["reply"],
            "grounded_on_top1": reply_result["grounded_on"][0] if reply_result["grounded_on"] else None,
            "top_similarity": reply_result["top_similarity"],
            "action": decision["action"],
            "action_reason": decision["reason"],
        }


if __name__ == "__main__":
    agent = SupportAgent()
    demo_msgs = [
        "cant sign into icloud on my iPhone after the update, keeps saying incorrect password",
        "just want to say the genius bar team fixed my phone in 20 mins, incredible service",
        "just got charged $9.99 for icloud storage i never signed up for???",
    ]
    for m in demo_msgs:
        result = agent.handle(m)
        print("-" * 80)
        for k, v in result.items():
            print(f"{k}: {v}")