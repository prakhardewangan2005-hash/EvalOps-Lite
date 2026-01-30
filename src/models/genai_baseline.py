from dataclasses import dataclass
from typing import Dict


@dataclass
class GenAIConfig:
    temperature: float = 0.7
    max_tokens: int = 256


class GenAIBaseline:
    """
    Lightweight GenAI-inspired baseline (no heavy deps).
    Produces score + explanation for demo/eval usage.
    """

    def __init__(self, config: GenAIConfig = GenAIConfig()):
        self.config = config

    def predict(self, text: str) -> Dict:
        score = self._heuristic_score(text)
        return {
            "input": text,
            "genai_score": score,
            "confidence": min(0.95, 0.45 + score),
            "explanation": self._explain(score),
            "model_type": "lightweight-genai-baseline",
        }

    def _heuristic_score(self, text: str) -> float:
        t = (text or "").strip().lower()
        length_factor = min(len(t) / 200, 1.0)

        positive = any(k in t for k in ["good", "excellent", "positive", "secure", "efficient"])
        negative = any(k in t for k in ["bad", "poor", "negative", "unsafe", "slow"])

        base = 0.25 + 0.55 * length_factor
        if positive:
            base += 0.2
        if negative:
            base -= 0.2

        return round(max(0.0, min(1.0, base)), 2)

    def _explain(self, score: float) -> str:
        if score >= 0.75:
            return "High semantic signal detected via lightweight reasoning heuristics."
        if score >= 0.45:
            return "Moderate semantic signal detected."
        return "Low semantic signal detected."
