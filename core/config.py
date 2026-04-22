"""
配置管理模块。
支持从环境变量、.env 文件或代码中直接设置。
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0          # 学术场景建议 0.0 保证严谨
    max_tokens: int = 4096
    timeout: int = 60

    def __post_init__(self):
        # 从环境变量自动读取 API Key
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "ollama":
                self.base_url = self.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    short_term_max_tokens: int = 10000
    # RAG 配置
    vector_db_path: str = "./data/vector_db"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class SkillsConfig:
    """技能系统配置"""
    skills_dir: str = "./skills"       # 自定义技能存放目录
    enable_builtin: bool = True
    # arXiv 配置
    arxiv_categories: list[str] = field(default_factory=lambda: ["hep-th", "hep-ph", "quant-ph", "gr-qc", "astro-ph.CO"])
    arxiv_max_results: int = 50


@dataclass
class AgentConfig:
    """Agent 全局配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    # 工作目录
    data_dir: str = "./data"
    log_level: str = "INFO"
    # 是否启用中间件（日志、指标追踪）
    enable_telemetry: bool = True
    enable_sandbox: bool = True

    def __post_init__(self):
        # 确保数据目录存在
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)


# 全局配置实例（可在运行时修改）
config = AgentConfig()


def get_config() -> AgentConfig:
    """获取全局配置"""
    return config


def update_config(**kwargs):
    """动态更新配置"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
