# packages/graphrag/src/graphrag/llm/loader.py
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from .base import LLM
from .hf_local import HFLocalLLM
from ..config import LLMConfig


class NoopLLM(LLM):
    def generate(self, prompt: str) -> str:
        return (
            "NOOP_LLM_RESPONSE\n"
            "This is a placeholder response because LLM_PROVIDER=noop.\n\n"
            "PROMPT:\n"
            + prompt[:4000]
        )


@dataclass
class HFChatResponse:
    content: str
    raw: dict[str, Any]


class HFInferenceLLM(LLM):
    """
    Hugging Face Router Inference (remote), OpenAI-compatible chat endpoint.

    Uses:
      POST https://router.huggingface.co/v1/chat/completions

    Required:
      - model_id (e.g., Qwen/Qwen2.5-7B-Instruct or *-Turbo)
      - token (HF_TOKEN)
    """

    def __init__(
        self,
        model_id: str,
        token: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        timeout_s: int = 120,
        verbose: bool = False,
    ) -> None:
        self.model_id = model_id
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_s = timeout_s
        self.verbose = verbose

        # ✅ HF Router (OpenAI-compatible)
        self.endpoint = "https://router.huggingface.co/v1/chat/completions"

    def _post(self, payload: dict[str, Any]) -> HFChatResponse:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Cloudflare / some providers block python-urllib UA fingerprints.
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw_bytes = resp.read()
                raw_text = raw_bytes.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            # Try to read response body for a better message
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            raise RuntimeError(
                f"HF Router error {e.code}: {e.reason}\n{err_body}".strip()
            ) from e
        except Exception as e:
            raise RuntimeError(f"HF Router request failed: {e}") from e

        try:
            obj = json.loads(raw_text)
        except Exception as e:
            raise RuntimeError(f"HF Router returned non-JSON:\n{raw_text[:2000]}") from e

        if self.verbose:
            print("HF Router raw response keys:", list(obj.keys()))

        # OpenAI-style: choices[0].message.content
        try:
            content = obj["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(
                f"HF Router JSON missing expected fields. Got keys={list(obj.keys())}\n"
                f"First 2000 chars:\n{raw_text[:2000]}"
            ) from e

        return HFChatResponse(content=content, raw=obj)

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            # HF expects max_tokens (OpenAI-style). We map from max_new_tokens.
            "max_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }
        res = self._post(payload)
        return res.content


def load_llm(cfg: LLMConfig) -> LLM:
    """
    Supported providers:

    - hf_inference: remote HF router (recommended for Mac laptop)
      env via config.py:
        LLM_PROVIDER=hf_inference
        HF_MODEL_ID=...
        HF_TOKEN=...

    - hf_local: local transformers (requires GPU/CPU RAM)
      env via config.py:
        LLM_PROVIDER=hf_local
        LLM_MODEL=...

    - noop: deterministic placeholder
      LLM_PROVIDER=noop
    """
    provider = (cfg.provider or "noop").strip().lower()

    if provider == "hf_local":
        if not cfg.model:
            raise ValueError("LLM_PROVIDER=hf_local requires LLM_MODEL to be set.")
        return HFLocalLLM(
            model_id=cfg.model,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

    if provider == "hf_inference":
        if not cfg.model:
            raise ValueError(
                "LLM_PROVIDER=hf_inference requires HF_MODEL_ID to be set "
                "(wired into cfg.model by AppConfig.from_env())."
            )
        if not cfg.token:
            raise ValueError("LLM_PROVIDER=hf_inference requires HF_TOKEN to be set.")
        return HFInferenceLLM(
            model_id=cfg.model,
            token=cfg.token,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

    return NoopLLM()

