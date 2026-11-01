"""Settings come from environment variables (GitHub Actions secrets / variables)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


@dataclass
class Config:
    groq_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "openai/gpt-oss-120b"))
    gemini_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-flash-lite-latest"))
    voice: str = field(default_factory=lambda: _env("KOKORO_VOICE", "am_adam"))
    speed: float = field(default_factory=lambda: float(_env("KOKORO_SPEED", "0.98")))
    model_dir: str = field(default_factory=lambda: _env("KOKORO_DIR", os.path.expanduser("~/.cache/kokoro")))
    ig_user_id: str = field(default_factory=lambda: _env("IG_USER_ID"))
    ig_token: str = field(default_factory=lambda: _env("IG_ACCESS_TOKEN"))
    graph_host: str = field(default_factory=lambda: _env("IG_GRAPH_HOST", "graph.instagram.com"))
    graph_version: str = field(default_factory=lambda: _env("IG_GRAPH_VERSION", "v23.0"))
    repo: str = field(default_factory=lambda: _env("GITHUB_REPOSITORY"))
    gh_token: str = field(default_factory=lambda: _env("GITHUB_TOKEN"))
    publish_mode: str = field(default_factory=lambda: _env("PUBLISH_MODE", "auto"))   # auto | dry
    character: str = "assets/puppet"

    @property
    def can_publish(self) -> bool:
        return (self.publish_mode != "dry" and bool(self.ig_user_id and self.ig_token
                and self.repo and self.gh_token))
