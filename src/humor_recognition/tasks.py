from .configs import *
from .dataset import HumorDataset
from .models import *
from .pipelines import MultimodalPipeline

import pandas as pd

from transformers import TrainingArguments, Trainer


def train(corpus_path,
          features_path,
          checkpoint,
          base_model_type,
          method,
          output):
    train_corpus = pd.read_json(corpus_path)
    train_features = pd.read_csv(features_path, index_col=0)
    train_features = train_features.drop(columns=['Label'])
    train_dataset = HumorDataset(train_corpus['Text'].values,
                                 train_features.values,
                                 train_corpus['Label'].values,
                                 feature_names=train_features.columns.values,
                                 tokenize=checkpoint)

    model_config = method.config_class(base_model=checkpoint,
                                       base_model_type=base_model_type,
                                       num_labels=2,
                                       num_features=train_dataset.shape[1],
                                       freeze_base=False,
                                       label2id=train_dataset.label2id,
                                       id2label=train_dataset.id2label,
                                       hidden_dim=786)
    model = method(model_config)

    training_args = TrainingArguments(output_dir=output,
                                      save_strategy='epoch',
                                      save_total_limit=1,
                                      learning_rate=5e-5,
                                      num_train_epochs=5,
                                      per_device_train_batch_size=16)
    trainer = Trainer(model, training_args,
                      train_dataset=train_dataset)
    trainer.train()
    return model


def predict(corpus_path,
            features_path,
            trained_model,
            output):
    test_corpus = pd.read_json(corpus_path)
    test_features = pd.read_csv(features_path, index_col=0)
    test_features = test_features.drop(columns=['Label'])
    test_dataset = HumorDataset(test_corpus['Text'].values,
                                test_features.values,
                                test_corpus['Label'].values,
                                feature_names=test_features.columns.values,
                                tokenize=trained_model.config.base_model,
                                label2id=trained_model.config.label2id)

    text_classification = MultimodalPipeline(model=trained_model, device=0)
    predictions = text_classification(test_dataset)

    results = pd.DataFrame(predictions, index=test_corpus.index)
    results = results.drop(columns='score')
    results = results.rename(columns={'label': 'Prediction'})
    results['Label'] = test_corpus['Label']

    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, encoding='utf-8')
