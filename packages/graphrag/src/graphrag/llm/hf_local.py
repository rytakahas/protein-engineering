from __future__ import annotations
from dataclasses import dataclass

from .base import LLM

# Optional dependency: transformers + torch
# pip install transformers torch
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:  # pragma: no cover
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


@dataclass
class HFLocalLLM(LLM):
    model_id: str
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.2

    def __post_init__(self):
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers/torch not installed. pip install transformers torch")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch and torch.cuda.is_available() else None,
            device_map="auto" if self.device == "auto" else None,
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

