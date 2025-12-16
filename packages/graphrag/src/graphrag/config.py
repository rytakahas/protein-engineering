from __future__ import annotations
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "noop"  # "hf_local" or "noop"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.2
    hf_token: str | None = None


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
        llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "noop"),
            model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            device=os.getenv("LLM_DEVICE", "auto"),
            max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "512")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            hf_token=os.getenv("HF_TOKEN", None),
        )
        return AppConfig(neo4j=neo4j, llm=llm)

