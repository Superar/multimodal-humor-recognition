from torch.utils.data import Dataset
from transformers import AutoTokenizer

import numpy as np
import torch


class HumorDataset(Dataset):
    def __init__(self, texts, features, labels,
                 feature_names=None,
                 tokenize=False,
                 label2id=None):
        super().__init__()
        self.texts = np.array(texts)
        self.features = np.array(features)
        self.feature_names = np.array(feature_names)

        self.label2id = ({label: id_ for id_, label in enumerate(set(labels))}
                         if label2id is None else label2id)
        self.id2label = {id_: label for label, id_ in self.label2id.items()}
        self.labels = np.array([self.label2id[label] for label in labels])

        self.bert_token_data = None
        if tokenize:
            self.bert_tokenize(tokenize)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        if self.bert_token_data is not None:
            return {'input_ids': self.bert_token_data['input_ids'][index],
                    'attention_mask': self.bert_token_data['attention_mask'][index],
                    'features': torch.tensor(self.features[index, :]),
                    'labels': torch.tensor(self.labels[index])}
        else:
            return {'text': self.texts[index],
                    'features': self.features[index],
                    'labels': self.labels[index]}

    @property
    def shape(self):
        return self.features.shape

    def bert_tokenize(self, model):
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenized_data = tokenizer(self.texts.tolist(),
                                   padding='longest',
                                   return_tensors='pt',
                                   return_attention_mask=True)
        self.bert_token_data = tokenized_data
