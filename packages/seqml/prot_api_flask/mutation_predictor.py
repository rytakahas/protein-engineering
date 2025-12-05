# mutation_predictor.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import re

# Load fine-tuned ProtBERT model (adjust path if needed)
MODEL_PATH = "./saved_model/protbert_mutation"  # Change to your real model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model = model.to(DEVICE)
model.eval()

def clean_fasta(fasta_str):
    """Extract raw sequence from FASTA string."""
    lines = fasta_str.strip().split("\n")
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    return re.sub(r'[^A-Z]', '', sequence)

def mutate_sequence(wt_seq, position, new_aa):
    """Introduce a single amino acid mutation."""
    pos = int(position) - 1  # 1-based indexing
    if pos < 0 or pos >= len(wt_seq):
        raise ValueError(f"Position {position} is out of range for sequence length {len(wt_seq)}")
    return wt_seq[:pos] + new_aa + wt_seq[pos + 1:]

def predict_from_fasta_and_experiment(wt_fasta, mutation_df=None):
    """Run inference using fine-tuned ProtBERT on given mutations."""
    wt_seq = clean_fasta(wt_fasta)
    predictions = []

    # If no mutation file is given, just predict for the WT
    if mutation_df is None or mutation_df.empty:
        mutated_seqs = [wt_seq]
    else:
        mutated_seqs = []
        for _, row in mutation_df.iterrows():
            pos = int(row["POSITION"])
            new_aa = str(row["SEQUENCE"]).strip()
            try:
                mutated = mutate_sequence(wt_seq, pos, new_aa)
                mutated_seqs.append(mutated)
            except Exception as e:
                predictions.append({
                    "position": pos,
                    "error": str(e)
                })

    # Run batch prediction
    for seq in mutated_seqs:
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.sigmoid(outputs.logits).item()  # adjust based on your output format

        predictions.append({
            "sequence": seq,
            "predicted_score": round(score, 4)
        })

    return predictions

