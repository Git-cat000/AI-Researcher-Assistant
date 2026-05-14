"""Configuration dataclasses for AI Researcher Assistant.

Configuration is explicit and lazy. Importing this module does not read `.env`
files, create directories, or open external resources.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

_config: "AgentConfig | None" = None


@dataclass
class LLMConfig:
    """LLM provider configuration.

    The `openai` provider is intentionally OpenAI-compatible: it can target
    OpenAI, Azure OpenAI, DeepSeek, OpenRouter, SiliconFlow, and other services
    that expose an OpenAI-style chat completions API through `base_url`.
    """

    provider: Literal[
        "openai",
        "openai_compatible",
        "azure_openai",
        "deepseek",
        "openrouter",
        "siliconflow",
        "qwen",
        "anthropic",
        "ollama",
        "local",
    ] = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 60

    def __post_init__(self) -> None:
        if self.api_key is None:
            if self.provider in {"openai", "openai_compatible", "azure_openai"}:
                self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            elif self.provider == "deepseek":
                self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            elif self.provider == "openrouter":
                self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            elif self.provider == "siliconflow":
                self.api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
            elif self.provider == "qwen":
                self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider in {"ollama", "local"}:
                self.base_url = self.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class MemoryConfig:
    """Memory and RAG configuration."""

    short_term_max_tokens: int = 10000
    vector_db_path: str = "./data/vector_db"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class SkillsConfig:
    """Skill system configuration."""

    skills_dir: str = "./skills"
    enable_builtin: bool = True
    arxiv_categories: list[str] = field(
        default_factory=lambda: ["hep-th", "hep-ph", "quant-ph", "gr-qc", "astro-ph.CO"]
    )
    arxiv_max_results: int = 50


@dataclass
class AgentConfig:
    """Top-level agent configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    data_dir: str = "./data"
    log_level: str = "INFO"
    enable_telemetry: bool = True
    enable_sandbox: bool = True


def get_config() -> AgentConfig:
    """Return the process default config, creating it lazily."""

    global _config
    if _config is None:
        _config = AgentConfig()
    return _config


def load_config_from_env(env_file: str | None = None) -> AgentConfig:
    """Explicitly load environment variables and return a fresh config."""

    load_dotenv(env_file)
    return AgentConfig()


def update_config(config: AgentConfig | None = None, **kwargs) -> AgentConfig:
    """Update a config object and make it the process default."""

    global _config
    config = config or get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    _config = config
    return config
