"""
Multi-LLM Ensemble System

Cross-validation and ensemble using multiple frontier LLMs:
- Claude Opus 4.5 (Anthropic) - Deep reasoning
- GPT-5.2 Thinking (OpenAI) - Multi-step reasoning
- DeepSeek-V3 (DeepSeek) - Cost-effective

Core principle: Different LLMs have different biases and blind spots. Ensemble can:
1. Reduce single-model hallucinations
2. Identify signals where multiple models agree (more reliable)
3. Detect disagreement points (uncertainty quantification)
"""

from .multi_llm_ensemble import (
    LLMProvider,
    LLMConfig,
    MultiLLMEnsemble,
    EnsembleResult,
)
from .llm_clients import (
    ClaudeClient,
    OpenAIClient,
    DeepSeekClient,
    BaseLLMClient,
)

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "MultiLLMEnsemble",
    "EnsembleResult",
    "ClaudeClient",
    "OpenAIClient",
    "DeepSeekClient",
    "BaseLLMClient",
]
