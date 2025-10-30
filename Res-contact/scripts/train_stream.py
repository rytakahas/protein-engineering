import yaml, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from rescontact.models.bilinear_scorer import BilinearScorer

# import the helpers you pasted in your script (or keep them inside this file)
# from rescontact.features.embedding import ESMEmbedderHF, ESMEmbedder
# from rescontact.features.msa import read_a3m_as_rows, read_fasta_as_rows, msa_1d_features
# from rescontact.data.pdb_utils import load_structure
# ... plus all functions you already have in your notebook (extractors, pairs, etc.)

from your_notebook_port import (   # ← create this module by pasting your long code, or paste it below directly
    collect_structs, try_load_esm2, train_stream, evaluate_stream
)

def main():
    with open("configs/rescontact.yaml") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    stream = cfg["stream"]
    CA_DIST = cfg["labels"]["contact_threshold_angstrom"]

    train_dir = Path(paths["train_dir"])
    test_dir  = Path(paths["test_dir"])
    train_files = collect_structs(train_dir)
    test_files  = collect_structs(test_dir)

    if stream["max_train_files"] and len(train_files) > stream["max_train_files"]:
        train_files = train_files[: stream["max_train_files"]]

    # Load HF ESM (or keep fair-esm if you prefer)
    tokenizer, mdl = try_load_esm2(stream["esm_hf_model"])

    tr, val = train_test_split(train_files, test_size=0.2, random_state=42)

    model = train_stream(tr, tokenizer, mdl,
                         epochs=cfg["training"]["epochs"],
                         lr=cfg["training"]["lr"],
                         rank=128)

    scores = evaluate_stream(val, tokenizer, mdl, model, split_name="val")
    print(scores)

    os.makedirs("checkpoints", exist_ok=True)
    import torch
    torch.save({"state_dict": model.state_dict()}, "checkpoints/rescontact_stream.pt")
    print("Saved checkpoints/rescontact_stream.pt")

if __name__ == "__main__":
    main()

