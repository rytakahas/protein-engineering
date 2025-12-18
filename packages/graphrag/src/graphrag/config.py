
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"


@dataclass(frozen=True)
class LLMConfig:
    # provider: hf_inference (HF Router) | hf_local | noop
    provider: str = "noop"
    model: str | None = None
    token: str | None = None
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.2
    timeout_s: int = 60


@dataclass(frozen=True)
class AppConfig:
    neo4j: Neo4jConfig
    llm: LLMConfig

    @staticmethod
    def from_env() -> "AppConfig":
        neo4j = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "neo4j"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )

        provider = os.getenv("LLM_PROVIDER", "noop").strip()

        model = None
        if provider == "hf_inference":
            model = os.getenv("HF_MODEL_ID")
        elif provider == "hf_local":
            model = os.getenv("LLM_MODEL")

        llm = LLMConfig(
            provider=provider,
            model=model,
            token=os.getenv("HF_TOKEN"),
            device=os.getenv("LLM_DEVICE", "auto"),
            max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "512")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            timeout_s=int(os.getenv("LLM_TIMEOUT_S", "60")),
        )

        return AppConfig(neo4j=neo4j, llm=llm)
