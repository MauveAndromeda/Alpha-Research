"""
Multi-LLM Ensemble System

使用多个前沿大模型进行交叉验证和集成:
- Claude Opus 4.5 (Anthropic) - 深度推理
- GPT-5.2 Thinking (OpenAI) - 多步推理
- DeepSeek-V3 (DeepSeek) - 成本效益

核心思路: 不同LLM有不同的bias和盲点，集成可以:
1. 减少单一模型的幻觉
2. 发现多模型一致的信号 (更可靠)
3. 识别分歧点 (不确定性量化)
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
