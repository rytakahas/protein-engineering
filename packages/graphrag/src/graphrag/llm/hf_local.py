from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .base import LLM
from ..config import LLMConfig

class HFLocalLLM(LLM):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if cfg.device == "auto" else None

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map=device_map
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            do_sample=self.cfg.temperature > 0,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()
