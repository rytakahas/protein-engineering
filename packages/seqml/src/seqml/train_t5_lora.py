
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


@dataclass
class TrainConfig:
    model_name: str = "google-t5/t5-small"
    lr: float = 5e-4
    epochs: int = 2
    batch_size: int = 8
    max_len: int = 512
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    output_dir: str = "outputs/t5"
    fp16: bool = False


def to_instruction(row) -> str:
    return f"Predict fitness for sequence: {row['mut_seq']}\\nAnswer:"


def train_from_csv(csv_path: str, cfg: TrainConfig):
    df = pd.read_csv(csv_path)
    assert "mut_seq" in df.columns and "label" in df.columns, "CSV must have mut_seq,label"
    df["text"] = df.apply(to_instruction, axis=1)
    df["labels"] = df["label"].astype(str)

    dset = Dataset.from_pandas(df[["text", "labels"]])
    tok = T5TokenizerFast.from_pretrained(cfg.model_name)
    mdl = T5ForConditionalGeneration.from_pretrained(cfg.model_name)

    if cfg.use_lora and PEFT_AVAILABLE:
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q", "v", "k", "o"],
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        mdl = get_peft_model(mdl, lora_cfg)

    def tok_fn(ex):
        enc = tok(
            ex["text"],
            max_length=cfg.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with tok.as_target_tokenizer():
            dec = tok(
                ex["labels"], max_length=32, truncation=True, padding="max_length", return_tensors="pt"
            )
        enc["labels"] = dec["input_ids"]
        return {k: v.squeeze(0) for k, v in enc.items()}

    tok_dset = dset.map(tok_fn)
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        logging_steps=50,
        save_steps=200,
        fp16=cfg.fp16,
        report_to=[],
    )

    trainer = Trainer(model=mdl, args=args, train_dataset=tok_dset)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir
