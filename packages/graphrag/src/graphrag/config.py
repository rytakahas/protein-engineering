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
    provider: str  # hf_inference | hf_local | none
    model: str | None = None
    token: str | None = None
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.2


@dataclass(frozen=True)
class AppConfig:
    neo4j: Neo4jConfig
    llm: LLMConfig

    @staticmethod
    def from_env() -> "AppConfig":
        neo4j = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )

        provider = os.getenv("LLM_PROVIDER", "hf_inference").strip()
        llm = LLMConfig(
            provider=provider,
            model=os.getenv("HF_MODEL_ID") if provider == "hf_inference" else os.getenv("LLM_MODEL"),
            token=os.getenv("HF_TOKEN"),
            device=os.getenv("LLM_DEVICE", "auto"),
            max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "512")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        return AppConfig(neo4j=neo4j, llm=llm)

