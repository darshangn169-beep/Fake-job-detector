"""OPTIONAL: fine-tune DistilBERT on the same split and compare against the TF-IDF model.

Meant for a free Google Colab GPU (Runtime > Change runtime type > T4 GPU).
    !pip install -q transformers datasets accelerate scikit-learn pandas
    !python train_bert.py

NOTE: this script was not run in the environment where the rest of the project was built
(no GPU / torch there). Run it yourself, check it works, and only then quote its numbers.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)

import jobscam

SEED, MODEL = 42, "distilbert-base-uncased"

df = pd.read_csv("data/fake_job_postings.csv")
y = df["fraudulent"].values
# SAME split as train.py so the comparison is fair
train_df, test_df, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)


def raw_text(frame):
    # BERT works better on lightly cleaned text, so keep case/punctuation
    return [" ".join(v for v in (r.get(f) for f in jobscam.TEXT_FIELDS) if isinstance(v, str))
            for r in frame.to_dict("records")]


tok = AutoTokenizer.from_pretrained(MODEL)


class DS(Dataset):
    def __init__(self, texts, labels):
        self.enc = tok(texts, truncation=True, max_length=256, padding="max_length")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


train_ds, test_ds = DS(raw_text(train_df), y_train), DS(raw_text(test_df), y_test)

# Class imbalance: weight the loss so the rare fake class matters
weights = torch.tensor([1.0, (y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float)


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=weights.to(out.logits.device))
        return (loss, out) if return_outputs else loss


model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
args = TrainingArguments(output_dir="bert_out", num_train_epochs=3, per_device_train_batch_size=16,
                         per_device_eval_batch_size=64, learning_rate=2e-5, fp16=torch.cuda.is_available(),
                         seed=SEED, report_to="none", save_strategy="no")
trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds)
trainer.train()

logits = trainer.predict(test_ds).predictions
proba = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
pred = (proba >= 0.5).astype(int)
print("DistilBERT test:",
      f"precision={precision_score(y_test, pred):.3f}",
      f"recall={recall_score(y_test, pred):.3f}",
      f"f1={f1_score(y_test, pred):.3f}",
      f"pr_auc={average_precision_score(y_test, proba):.3f}")
