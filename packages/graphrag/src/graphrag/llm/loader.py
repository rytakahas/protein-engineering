
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict

from .base import LLM
from .hf_local import HFLocalLLM
from ..config import LLMConfig


class NoopLLM(LLM):
    def generate(self, prompt: str) -> str:
        return (
            "NOOP_LLM_RESPONSE\n"
            "LLM_PROVIDER=noop so no remote/local model is being called.\n\n"
            "PROMPT_PREVIEW:\n"
            + prompt[:4000]
        )


@dataclass
class HFRouterLLM(LLM):
    """
    Hugging Face Router Chat Completions API:
      POST https://router.huggingface.co/v1/chat/completions
    """

    model: str
    token: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    timeout_s: int = 60
    endpoint: str = "https://router.huggingface.co/v1/chat/completions"

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HF Router error {e.code}: {body}") from e

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
        }
        res = self._post(payload)

        try:
            return res["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(res, indent=2, ensure_ascii=False)


def load_llm(cfg: LLMConfig) -> LLM:
    provider = (cfg.provider or "noop").strip()

    if provider == "hf_inference":
        if not cfg.model:
            raise ValueError("LLM_PROVIDER=hf_inference requires HF_MODEL_ID to be set (mapped to cfg.model).")
        if not cfg.token:
            raise ValueError("LLM_PROVIDER=hf_inference requires HF_TOKEN.")
        return HFRouterLLM(
            model=cfg.model,
            token=cfg.token,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            timeout_s=cfg.timeout_s,
        )

    if provider == "hf_local":
        if not cfg.model:
            raise ValueError("LLM_PROVIDER=hf_local requires LLM_MODEL.")
        return HFLocalLLM(
            model_id=cfg.model,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

    return NoopLLM()
