"""
LLM Client Implementations

支持多个前沿大模型:
- Claude Opus 4.5 (claude-opus-4-5-20251101)
- GPT-5.2 with Thinking (gpt-5.2-thinking)
- DeepSeek-V3 (deepseek-chat)
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM响应结构"""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]  # tokens
    latency_ms: float
    raw_response: Optional[Dict] = None
    thinking: Optional[str] = None  # For models with thinking capability


class BaseLLMClient(ABC):
    """LLM客户端基类"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._request_count = 0
        self._total_tokens = 0

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """生成响应"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """获取模型名称"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """获取提供商名称"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
        }


class ClaudeClient(BaseLLMClient):
    """
    Claude Opus 4.5 Client

    特点:
    - 最强的深度推理能力
    - 优秀的金融分析能力
    - 支持长上下文 (200K tokens)
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-opus-4-5-20251101"
        self.base_url = "https://api.anthropic.com/v1/messages"

    def get_model_name(self) -> str:
        return self.model

    def get_provider_name(self) -> str:
        return "Anthropic"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """调用Claude API"""
        import aiohttp
        import time

        start_time = time.time()

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url, headers=headers, json=payload
                ) as response:
                    result = await response.json()

                    if response.status != 200:
                        raise Exception(f"Claude API error: {result}")

                    content = result["content"][0]["text"]
                    usage = result.get("usage", {})

                    self._request_count += 1
                    self._total_tokens += usage.get("input_tokens", 0) + usage.get(
                        "output_tokens", 0
                    )

                    return LLMResponse(
                        content=content,
                        model=self.model,
                        provider="Anthropic",
                        usage={
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                        },
                        latency_ms=(time.time() - start_time) * 1000,
                        raw_response=result,
                    )

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise


class OpenAIClient(BaseLLMClient):
    """
    GPT-5.2 Thinking Client

    特点:
    - 内置思考链 (Chain of Thought)
    - 多步推理能力强
    - 适合复杂分析任务
    """

    def __init__(self, api_key: Optional[str] = None, use_thinking: bool = True):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"))
        # 使用thinking模型或普通模型
        self.model = "o3" if use_thinking else "gpt-4.1"
        self.use_thinking = use_thinking
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def get_model_name(self) -> str:
        return self.model

    def get_provider_name(self) -> str:
        return "OpenAI"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """调用OpenAI API"""
        import aiohttp
        import time

        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }

        # Thinking models use different temperature handling
        if not self.use_thinking:
            payload["temperature"] = temperature

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url, headers=headers, json=payload
                ) as response:
                    result = await response.json()

                    if response.status != 200:
                        raise Exception(f"OpenAI API error: {result}")

                    choice = result["choices"][0]
                    content = choice["message"]["content"]
                    usage = result.get("usage", {})

                    # Extract thinking if available
                    thinking = None
                    if self.use_thinking and "reasoning_content" in choice["message"]:
                        thinking = choice["message"]["reasoning_content"]

                    self._request_count += 1
                    self._total_tokens += usage.get("total_tokens", 0)

                    return LLMResponse(
                        content=content,
                        model=self.model,
                        provider="OpenAI",
                        usage={
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        },
                        latency_ms=(time.time() - start_time) * 1000,
                        raw_response=result,
                        thinking=thinking,
                    )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek-V3 Client

    特点:
    - 极高的性价比 (比Claude便宜10x)
    - 中文能力强
    - 适合批量分析任务
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("DEEPSEEK_API_KEY"))
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def get_model_name(self) -> str:
        return self.model

    def get_provider_name(self) -> str:
        return "DeepSeek"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """调用DeepSeek API"""
        import aiohttp
        import time

        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url, headers=headers, json=payload
                ) as response:
                    result = await response.json()

                    if response.status != 200:
                        raise Exception(f"DeepSeek API error: {result}")

                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})

                    self._request_count += 1
                    self._total_tokens += usage.get("total_tokens", 0)

                    return LLMResponse(
                        content=content,
                        model=self.model,
                        provider="DeepSeek",
                        usage={
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        },
                        latency_ms=(time.time() - start_time) * 1000,
                        raw_response=result,
                    )

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise


# 便捷函数
def create_llm_client(provider: str, **kwargs) -> BaseLLMClient:
    """创建LLM客户端"""
    providers = {
        "claude": ClaudeClient,
        "anthropic": ClaudeClient,
        "openai": OpenAIClient,
        "gpt": OpenAIClient,
        "deepseek": DeepSeekClient,
    }

    provider_lower = provider.lower()
    if provider_lower not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")

    return providers[provider_lower](**kwargs)
