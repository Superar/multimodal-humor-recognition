import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments
    )
    import marimo as mo
    import wandb

    from humor_recognition.configs import GeneralConfig, SharedRepresentationConfig
    from humor_recognition.dataset import HumorDataset
    from humor_recognition.models import (
        ClassificationModelConcatenation,
        ClassificationModelFeaturesPooling,
        ClassificationModelSharedRepresentation,
    )

    return (
        ClassificationModelConcatenation,
        ClassificationModelFeaturesPooling,
        ClassificationModelSharedRepresentation,
        HumorDataset,
        Path,
        Trainer,
        TrainingArguments,
        accuracy_score,
        f1_score,
        mo,
        np,
        pd,
        train_test_split,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Humor Classification: Multimodal Hyperparameter Tuning

    This notebook performs hyperparameter tuning for multimodal humor recognition models using **Weights & Biases Sweeps** in a **Marimo** reactive notebook environment.
    """)
    return


@app.cell
def _(Path):
    # Defaulting to paths inside the repository
    train_corpus_path = Path('data/cross_validation/fold_0/train.json')
    train_features_path = Path('data/humor_features/fold_0/train/data.hdf5')
    test_corpus_path = Path('data/cross_validation/fold_0/test.json')
    test_features_path = Path('data/humor_features/fold_0/test/data.hdf5')
    return (
        test_corpus_path,
        test_features_path,
        train_corpus_path,
        train_features_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration Settings
    """)
    return


@app.cell
def _(mo):
    model_dropdown = mo.ui.dropdown(
        options={
            "BERTimbau Base (neuralmind/bert-base-portuguese-cased)": "neuralmind/bert-base-portuguese-cased",
            "BERTimbau Large (neuralmind/bert-large-portuguese-cased)": "neuralmind/bert-large-portuguese-cased",
            "Albertina PT-BR (PORTULAN/albertina-900m-portuguese-ptpt-encoder)": "PORTULAN/albertina-900m-portuguese-ptpt-encoder",
            "Albertina PT-PT (PORTULAN/albertina-900m-portuguese-ptbr-encoder)": "PORTULAN/albertina-900m-portuguese-ptpt-encoder"
        },
        value="BERTimbau Base (neuralmind/bert-base-portuguese-cased)",
        label="Base Model"
    )

    model_type_radio = mo.ui.radio(
        options=["bert", "deberta"],
        value="bert",
        label="Base Model Type"
    )

    method_dropdown = mo.ui.dropdown(
        options={
            "Concatenation (x || n)": "concatenation",
            "Features Pooling (x || MLP(n))": "pooling",
            "Shared Representation (MLP(x || MLP(n)))": "shared"
        },
        value="Shared Representation (MLP(x || MLP(n)))",
        label="Fusion Method"
    )

    num_runs_slider = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=5,
        label="Number of Tuning Runs"
    )
    return method_dropdown, model_dropdown, model_type_radio, num_runs_slider


@app.cell(hide_code=True)
def _(method_dropdown, mo, model_dropdown, model_type_radio, num_runs_slider):
    mo.vstack([
        model_dropdown,
        model_type_radio,
        method_dropdown,
        num_runs_slider
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Loading
    """)
    return


@app.cell
def _(
    mo,
    pd,
    test_corpus_path,
    test_features_path,
    train_corpus_path,
    train_features_path,
):
    train_corpus = pd.read_json(train_corpus_path)
    train_features = pd.read_hdf(train_features_path, index_col=0)
    train_features = train_features.drop(columns=['Label'], errors='ignore')

    test_corpus = pd.read_json(test_corpus_path)
    test_features = pd.read_hdf(test_features_path, index_col=0)
    test_features = test_features.drop(columns=['Label'], errors='ignore')

    mo.md(
        '**Data Loaded successfully**\n'    
        f'- Shape of training corpus: {train_corpus.shape}\n'
        f'- Shape of training features: {train_features.shape}\n'
        f'- Shape of test corpus: {test_corpus.shape}\n'
        f'- Shape of test features: {test_features.shape}\n'
    )
    return train_corpus, train_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train/Validation Data Splitting
    """)
    return


@app.cell
def _(mo, train_corpus, train_features, train_test_split):
    train_idx, val_idx = train_test_split(
        range(len(train_corpus)),
        test_size=0.2,
        random_state=42,
        stratify=train_corpus['Label']
    )

    train_corpus_split = train_corpus.iloc[train_idx].reset_index(drop=True)
    val_corpus_split = train_corpus.iloc[val_idx].reset_index(drop=True)

    train_features_split = train_features.iloc[train_idx].reset_index(drop=True)
    val_features_split = train_features.iloc[val_idx].reset_index(drop=True)

    mo.md(
        '**Validation Split Done**\n'
        f'- New training corpus size: {len(train_corpus_split)}\n'
        f'- Validation corpus size: {len(val_corpus_split)}'
    )
    return (
        train_corpus_split,
        train_features_split,
        val_corpus_split,
        val_features_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tuning Execution Function
    """)
    return


@app.cell
def _(
    ClassificationModelConcatenation,
    ClassificationModelFeaturesPooling,
    ClassificationModelSharedRepresentation,
    HumorDataset,
    Trainer,
    TrainingArguments,
    accuracy_score,
    f1_score,
    method_dropdown,
    model_dropdown,
    model_type_radio,
    np,
    train_corpus_split,
    train_features_split,
    val_corpus_split,
    val_features_split,
    wandb,
):
    def run_sweep_training():
        run = wandb.init()

        # Hyperparameters chosen by the sweep
        lr = wandb.config.learning_rate
        batch_size = wandb.config.batch_size
        epochs = wandb.config.epochs
        hidden_dim = getattr(wandb.config, 'hidden_dim', 786)
        freeze_base = getattr(wandb.config, 'freeze_base', False)

        base_model = model_dropdown.value
        base_model_type = model_type_radio.value
        method_name = method_dropdown.value

        # Create datasets
        train_dataset = HumorDataset(
            train_corpus_split['Text'].values,
            train_features_split.values,
            train_corpus_split['Label'].values,
            feature_names=train_features_split.columns.values,
            tokenize=base_model
        )

        val_dataset = HumorDataset(
            val_corpus_split['Text'].values,
            val_features_split.values,
            val_corpus_split['Label'].values,
            feature_names=val_features_split.columns.values,
            tokenize=base_model,
            label2id=train_dataset.label2id
        )

        # Select Model Class
        if method_name == 'concatenation':
            method_class = ClassificationModelConcatenation
        elif method_name == 'pooling':
            method_class = ClassificationModelFeaturesPooling
        elif method_name == 'shared':
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
            hidden_dim=hidden_dim
        )
        model = method_class(model_config)

        # Compute metrics function for HF Trainer
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='macro')
            return {
                'accuracy': acc,
                'f1_macro': f1
            }

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./sweep_results',
            save_strategy='no',
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy='epoch',
            logging_strategy='epoch',
            report_to='wandb',
            disable_tqdm=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        run.finish()

    return (run_sweep_training,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run Sweep Control
    """)
    return


@app.cell
def _(mo):
    # Sweep configuration definition
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'eval/f1_macro',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 2,
            'eta': 2
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            'batch_size': {
                'values': [8, 16, 32]
            },
            'epochs': {
                'values': [3, 5]
            },
            'hidden_dim': {
                'values': [256, 512, 768]
            },
            'freeze_base': {
                'values': [False, True]
            }
        }
    }

    # Button to start the sweep
    run_sweep_button = mo.ui.run_button(
        label="Run Hyperparameter Sweep",
        tooltip="Click to start the sweep agent in the selected mode"
    )
    run_sweep_button
    return run_sweep_button, sweep_config


@app.cell
def _(
    mo,
    num_runs_slider,
    run_sweep_button,
    run_sweep_training,
    sweep_config,
    wandb,
):
    if run_sweep_button.value:
        mo.output.append(mo.md("Initializing sweep"))
        try:
            sweep_id = wandb.sweep(
                sweep=sweep_config,
                project="multimodal-humor-recognition",
                entity="limma_"
            )

            mo.output.append(mo.md(f"Sweep ID: `{sweep_id}`"))

            wandb.agent(
                sweep_id=sweep_id,
                function=run_sweep_training,
                count=num_runs_slider.value
            )

            mo.output.append(mo.md("**Sweep completed**"))
        except Exception as e:
            mo.output.append(mo.md(f"**Sweep failed**: {str(e)}"))
    return


if __name__ == "__main__":
    app.run()
