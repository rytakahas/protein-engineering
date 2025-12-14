from ..config import LLMConfig
from .hf_local import HFLocalLLM

def load_llm(cfg: LLMConfig):
    if cfg.provider == "hf_local":
        return HFLocalLLM(cfg)
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")
