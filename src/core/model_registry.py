from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.models.genai_baseline import GenAIBaseline, GenAIConfig


@dataclass
class ModelRegistry:
    """
    Central registry for models/pipelines.
    Keeping this outside src.api.* prevents circular imports.
    """
    genai: GenAIBaseline

    @classmethod
    def build(cls) -> "ModelRegistry":
        genai = GenAIBaseline(GenAIConfig())
        return cls(genai=genai)

    def info(self) -> Dict[str, Any]:
        return {
            "genai": {"type": "lightweight-genai-baseline"}
        }


# Singleton registry
model_registry: ModelRegistry = ModelRegistry.build()
