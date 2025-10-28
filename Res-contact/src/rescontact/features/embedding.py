
from typing import Optional, Tuple
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

class FallbackEmbedder(torch.nn.Module):
    """Tiny learned embedding for smoke tests when transformers isn't available."""
    def __init__(self, dim: int = 64, vocab: int = 26):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        torch.nn.init.xavier_uniform_(self.emb.weight)

    @torch.no_grad()
    def forward(self, seq: str) -> torch.Tensor:
        idx = torch.tensor([(ord(ch) % 26) for ch in seq], dtype=torch.long, device=DEVICE)
        return self.emb(idx)

def try_load_esm2(model_id: str) -> Tuple[Optional[object], torch.nn.Module]:
    """Try to load ESM2 from Transformers; if unavailable, return (None, FallbackEmbedder)."""
    try:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=False)
        mdl = AutoModel.from_pretrained(model_id, trust_remote_code=True, local_files_only=False)
        mdl.to(DEVICE).eval()
        return tok, mdl
    except Exception as e:
        print(f"[warn] Could not load {model_id}: {e}\nUsing FallbackEmbedder.")
        return None, FallbackEmbedder().to(DEVICE).eval()

@torch.no_grad()
def embed_sequence(seq: str, tokenizer, model) -> torch.Tensor:
    if tokenizer is None:
        return model(seq)
    tokens = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    out = model(**tokens)
    hidden = out.last_hidden_state[0]
    # remove special tokens if present
    if hidden.shape[0] >= len(seq) + 2:
        hidden = hidden[1:1+len(seq)]
    return hidden.detach()
