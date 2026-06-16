import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import functools
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import wandb

from humor_recognition.dataset import HumorDataset
from humor_recognition.models import (
    ClassificationModelConcatenation,
    ClassificationModelFeaturesPooling,
    ClassificationModelSharedRepresentation,
)

train_corpus_path = Path("data/cross_validation/fold_0/train.json")
train_features_path = Path("data/humor_features/fold_0/train/data.hdf5")

train_corpus = pd.read_json(train_corpus_path)
train_features = pd.read_hdf(train_features_path, index_col=0)
train_features = train_features.drop(columns=["Label"], errors="ignore")

# Train/Val split
train_idx, val_idx = train_test_split(
    range(len(train_corpus)),
    test_size=0.2,
    random_state=42,
    stratify=train_corpus["Label"],
)

train_corpus_split = train_corpus.iloc[train_idx].reset_index(drop=True)
val_corpus_split = train_corpus.iloc[val_idx].reset_index(drop=True)

train_features_split = train_features.iloc[train_idx].reset_index(drop=True)
val_features_split = train_features.iloc[val_idx].reset_index(drop=True)

print(f"Train size: {len(train_corpus_split)}, Val size: {len(val_corpus_split)}")

models = [
    ("neuralmind/bert-base-portuguese-cased", "bert"),
    ("neuralmind/bert-large-portuguese-cased", "bert"),
    ("PORTULAN/albertina-900m-portuguese-ptbr-encoder", "deberta"),
    ("PORTULAN/albertina-900m-portuguese-ptpt-encoder", "deberta"),
]

methods = ["concatenation", "pooling", "shared"]

NUM_RUNS_PER_SWEEP = 40
SWEEP_PROJECT = "multimodal-humor-recognition"
ENTITY = "limma_"


def get_sweep_config(base_model, method_name):
    model_name = base_model.split("/")[-1]
    return {
        "name": f"{model_name}-{method_name}",
        "method": "bayes",
        "metric": {"name": "eval/f1_macro", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 2, "eta": 2},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3,
            },
            "batch_size": {"values": [8, 16, 32]},
            "epochs": {"values": [5, 10, 20]},
            "hidden_dim": {"values": [256, 512, 768]},
            "freeze_base": {"values": [False, True]},
        },
    }


def run_sweep_training(
    base_model,
    base_model_type,
    method_name,
    train_corpus_split,
    train_features_split,
    val_corpus_split,
    val_features_split,
):
    run = wandb.init()

    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    hidden_dim = getattr(wandb.config, "hidden_dim", 786)
    freeze_base = getattr(wandb.config, "freeze_base", False)

    train_dataset = HumorDataset(
        train_corpus_split["Text"].values,
        train_features_split.values,
        train_corpus_split["Label"].values,
        feature_names=train_features_split.columns.values,
        tokenize=base_model,
    )

    val_dataset = HumorDataset(
        val_corpus_split["Text"].values,
        val_features_split.values,
        val_corpus_split["Label"].values,
        feature_names=val_features_split.columns.values,
        tokenize=base_model,
        label2id=train_dataset.label2id,
    )

    if method_name == "concatenation":
        method_class = ClassificationModelConcatenation
    elif method_name == "pooling":
        method_class = ClassificationModelFeaturesPooling
    elif method_name == "shared":
        method_class = ClassificationModelSharedRepresentation
    else:
        raise ValueError(f"Unknown method {method_name}")

    model_config = method_class.config_class(
        base_model=base_model,
        base_model_type=base_model_type,
        num_labels=2,
        num_features=train_dataset.shape[1],
        freeze_base=freeze_base,
        label2id=train_dataset.label2id,
        id2label=train_dataset.id2label,
        hidden_dim=hidden_dim,
    )
    model = method_class(model_config)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    model_short_name = base_model.split("/")[-1]
    output_dir = f"./results/sweeps/{model_short_name}_{method_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        logging_strategy="epoch",
        report_to="wandb",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    run.finish()


for base_model, base_model_type in models:
    for method_name in methods:
        print(f"Running {base_model} | {method_name}")

        sweep_config = get_sweep_config(base_model, method_name)

        sweep_id = wandb.sweep(sweep=sweep_config, project=SWEEP_PROJECT, entity=ENTITY)

        agent_func = functools.partial(
            run_sweep_training,
            base_model=base_model,
            base_model_type=base_model_type,
            method_name=method_name,
            train_corpus_split=train_corpus_split,
            train_features_split=train_features_split,
            val_corpus_split=val_corpus_split,
            val_features_split=val_features_split,
        )

        wandb.agent(sweep_id=sweep_id, function=agent_func, count=NUM_RUNS_PER_SWEEP)
