i# 🧬 ProtBERT Mutation Prediction API (Flask)

This is a REST API for predicting mutation effects using a fine-tuned ProtBERT model (see `notebook/protBert_ACT.ipynb`).

---

## 📁 Files

```yaml
prot_api_flask/
├── prot_api_flask.py # Flask app serving mutation predictions
├── mutation_predictor.py # Inference logic using fine-tuned ProtBERT
├── sample_input.json # Example request with WT + mutation data
└── README.md # This file

```

---

## 🧠 Functionality Overview

This API replicates and extends the logic in `notebook/protBert_ACT.ipynb`, allowing programmatic access to mutation effect predictions from:

1. A **wild-type FASTA string** (or plain sequence string)
2. An optional **experimental mutation table** (CSV or TXT), base64-encoded

---

## How to Run

```bash
pip install flask pandas
python prot_api_flask.py
```

Server runs on http://127.0.0.1:5050/mutation_prediction


## Input Format (JSON)

```json
{
  "wild_type_fasta": ">HeaderLine\\nMKTIIALSYIFCLVFADYKDDDDK",
  "experimental_csv_base64": "<base64-encoded .csv or .txt mutation table>"
}
```

wild_type_fasta: A FASTA-format or plain protein sequence string.

experimental_csv_base64: Optional. Base64-encoded tabular file (e.g., CSV, TSV, TXT).

Tabular file should include columns like:

``python-repl
POSITION,SEQUENCE,FITNESS
10,M,0.95
25,W,0.73
...


## Output Format (JSON)

```json
{
  "wild_type_length": 25,
  "num_experimental_mutants": 3,
  "predictions": [
    {
      "sequence": "GKTIIALSYIFCLVFADYKDDDDK",
      "predicted_score": 0.9342
    },
    {
      "sequence": "MKTIISLSYIFCLVFADYKDDDDK",
      "predicted_score": 0.8721
    },
    ...
  ]
}
```

##  Model Details (mutation_predictor.py)
Loads tokenizer and model from ./saved_model/protbert_mutation/

Handles:

Wild-type FASTA parsing

Mutation injection

Tokenization + inference using Hugging Face Transformers

Outputs sigmoid-scaled fitness scores (can be adjusted if model type differs)


🔄 Extensibility
This API is model-agnostic and can be adapted to:

Support multiple input formats (TSV, Excel)

Serve UniRep-based models (notebook/unirep_ACT.ipynb)

Add file upload support instead of base64

Handle multi-protein batch inference

🧬 Relation to Notebooks
Notebook	Purpose
notebook/protBert_ACT.ipynb	Fine-tune ProtBERT for activity prediction
notebook/unirep_ACT.ipynb	Fine-tune UniRep on mutation datasets
prot_api_flask/prot_api_flask.py	Deploy ProtBERT logic as a REST API

📄 License
MIT License – free for research and commercial use.


