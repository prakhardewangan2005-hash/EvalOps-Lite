from dataclasses import dataclass
from typing import Dict

@dataclass
class GenAIConfig:
    temperature: float = 0.7
    max_tokens: int = 256


class GenAIBaseline:
    """
    Lightweight GenAI-style baseline.
    Simulates LLM reasoning without heavy model dependencies.
    Production-safe for free-tier deployment.
    """

    def __init__(self, config: GenAIConfig = GenAIConfig()):
        self.config = config

    def predict(self, text: str) -> Dict:
        score = self._heuristic_score(text)

        return {
            "input": text,
            "genai_score": score,
            "confidence": min(0.95, 0.5 + score),
            "explanation": self._explain(score),
            "model_type": "lightweight-genai-baseline"
        }

    def _heuristic_score(self, text: str) -> float:
        length_factor = min(len(text) / 200, 1.0)
        keyword_factor = any(
            kw in text.lower()
            for kw in ["good", "excellent", "positive", "secure", "efficient"]
        )
        return round(0.4 * length_factor + (0.6 if keyword_factor else 0.2), 2)

    def _explain(self, score: float) -> str:
        if score > 0.7:
            return "High-quality semantic signal detected via heuristic reasoning."
        elif score > 0.4:
            return "Moderate semantic alignment detected."
        return "Low semantic alignment detected."
