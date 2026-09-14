"""Environment configuration.

Import this module first: it loads ``.env`` before anything imports Langfuse,
whose client is a process-wide singleton that reads its keys on first use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_app_title: str
    openrouter_app_url: str
    cube_url: str
    cube_api_secret: str
    as_of: date
    host: str
    port: int

    @property
    def langfuse_enabled(self) -> bool:
        keys = os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
        flag = os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() != "false"
        return bool(keys) and flag


def load_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        openrouter_app_title=os.environ.get("OPENROUTER_APP_TITLE", "ff-analytics-agent"),
        openrouter_app_url=os.environ.get("OPENROUTER_APP_URL", "https://github.com"),
        cube_url=os.environ.get("CUBE_URL", "http://localhost:4000").rstrip("/"),
        cube_api_secret=os.environ.get("CUBEJS_API_SECRET", ""),
        as_of=date.fromisoformat(os.environ.get("AGENT_AS_OF_DATE", "2026-09-14")),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )


settings = load_settings()
