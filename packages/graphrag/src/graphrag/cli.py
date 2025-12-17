# packages/graphrag/src/graphrag/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _getenv(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default


def _getenv_int(key: str, default: int) -> int:
    v = _getenv(key, None)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _getenv_float(key: str, default: float) -> float:
    v = _getenv(key, None)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _getenv_bool(key: str, default: bool = False) -> bool:
    v = _getenv(key, None)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Neo4jConfig:
    """Neo4j connection settings."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    # Optional: encrypted/ssl options (future)
    encrypted: bool = False


@dataclass(frozen=True)
class LLMConfig:
    """
    LLM settings.

    provider:
      - "hf_inference": Hugging Face Inference API (remote, laptop-friendly)
      - "hf_local":     Local transformers pipeline (heavier)
      - "none":         Disable LLM (retrieval-only)
    """
    provider: str = "hf_inference"

    # For hf_inference (remote)
    hf_model_id: Optional[str] = None
    hf_token: Optional[str] = None
    hf_endpoint: Optional[str] = None  # Optional custom endpoint

    # For hf_local (local)
    local_model_id: Optional[str] = None
    device: str = "auto"               # auto|cpu|cuda|mps
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Generation controls (used by both where applicable)
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95

    # Behavior
    verbose: bool = False


@dataclass(frozen=True)
class AppConfig:
    neo4j: Neo4jConfig
    llm: LLMConfig

    @staticmethod
    def from_env() -> "AppConfig":
        """
        Build config from environment variables (optionally loaded from .env via python-dotenv).

        Neo4j:
          NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, NEO4J_ENCRYPTED

        LLM:
          LLM_PROVIDER = hf_inference | hf_local | none

          HF Inference API:
            HF_MODEL_ID, HF_TOKEN, HF_ENDPOINT (optional)

          Local HF:
            LLM_MODEL (or HF_MODEL_ID), LLM_DEVICE, LLM_LOAD_IN_8BIT, LLM_LOAD_IN_4BIT

          Generation:
            LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LLM_TOP_P
        """
        neo4j = Neo4jConfig(
            uri=_getenv("NEO4J_URI", "bolt://localhost:7687") or "bolt://localhost:7687",
            user=_getenv("NEO4J_USER", "neo4j") or "neo4j",
            password=_getenv("NEO4J_PASSWORD", "password") or "password",
            database=_getenv("NEO4J_DATABASE", "neo4j") or "neo4j",
            encrypted=_getenv_bool("NEO4J_ENCRYPTED", False),
        )

        provider = (_getenv("LLM_PROVIDER", "hf_inference") or "hf_inference").lower()

        # Remote HF inference settings
        hf_model_id = _getenv("HF_MODEL_ID", None)
        hf_token = _getenv("HF_TOKEN", None)
        hf_endpoint = _getenv("HF_ENDPOINT", None)

        # Local model settings
        local_model_id = _getenv("LLM_MODEL", None) or hf_model_id  # allow reusing HF_MODEL_ID
        device = _getenv("LLM_DEVICE", "auto") or "auto"
        load_in_8bit = _getenv_bool("LLM_LOAD_IN_8BIT", False)
        load_in_4bit = _getenv_bool("LLM_LOAD_IN_4BIT", False)

        # Generation controls
        max_new_tokens = _getenv_int("LLM_MAX_NEW_TOKENS", 512)
        temperature = _getenv_float("LLM_TEMPERATURE", 0.2)
        top_p = _getenv_float("LLM_TOP_P", 0.95)

        llm = LLMConfig(
            provider=provider,
            hf_model_id=hf_model_id,
            hf_token=hf_token,
            hf_endpoint=hf_endpoint,
            local_model_id=local_model_id,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=_getenv_bool("LLM_VERBOSE", False),
        )

        return AppConfig(neo4j=neo4j, llm=llm)

    def validate(self) -> None:
        """
        Optional sanity checks. Call from CLI if you want friendlier errors.
        """
        if not self.neo4j.uri:
            raise ValueError("NEO4J_URI is required")
        if self.llm.provider not in {"hf_inference", "hf_local", "none"}:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.llm.provider}")

        if self.llm.provider == "hf_inference":
            if not self.llm.hf_model_id:
                raise ValueError("HF_MODEL_ID is required for LLM_PROVIDER=hf_inference")
            if not self.llm.hf_token:
                raise ValueError("HF_TOKEN is required for LLM_PROVIDER=hf_inference (set in .env)")

        if self.llm.provider == "hf_local":
            if not self.llm.local_model_id:
                raise ValueError("LLM_MODEL (or HF_MODEL_ID) is required for LLM_PROVIDER=hf_local")

