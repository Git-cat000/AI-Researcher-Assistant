"""LLM 工厂函数"""
from typing import Optional
from AI-Researcher-Assistant.core.config import LLMConfig, get_config
from AI-Researcher-Assistant.llm.base import BaseLLM
from AI-Researcher-Assistant.llm.openai import OpenAILLM
from AI-Researcher-Assistant.llm.anthropic import AnthropicLLM
from AI-Researcher-Assistant.llm.local import OllamaLLM
from AI-Researcher-Assistant.core.exceptions import ConfigurationError


def create_llm(config: Optional[LLMConfig] = None) -> BaseLLM:
    """
    根据配置创建 LLM 实例。
    
    Args:
        config: LLM 配置，若不提供则使用全局配置
        
    Returns:
        BaseLLM 实例
    """
    if config is None:
        config = get_config().llm
    
    provider = config.provider.lower()
    
    if provider == "openai":
        return OpenAILLM(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
    elif provider == "anthropic":
        return AnthropicLLM(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif provider in ["ollama", "local"]:
        return OllamaLLM(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    else:
        raise ConfigurationError(f"Unsupported LLM provider: {provider}")


def get_llm() -> BaseLLM:
    """获取默认 LLM 实例（使用全局配置）"""
    return create_llm()
