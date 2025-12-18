
from __future__ import annotations

from .base import LLM


class HFLocalLLM(LLM):
    def __init__(self, model_id: str, device: str = "auto", max_new_tokens: int = 512, temperature: float = 0.2):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
        )
        return out[0]["generated_text"]
