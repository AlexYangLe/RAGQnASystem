"""当前 LightRAG 问答链路的运行配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


@dataclass(frozen=True)
class Settings:
    """通过环境变量覆盖的本地运行时配置。"""

    OLLAMA_QWEN_MODEL: str
    LOG_LEVEL: str


settings = Settings(
    OLLAMA_QWEN_MODEL=_env("OLLAMA_QWEN_MODEL", "qwen2.5:7b"),
    LOG_LEVEL=_env("LOG_LEVEL", "INFO"),
)
