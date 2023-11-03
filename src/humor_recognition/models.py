import torch

from .configs import *

from torch import nn
from transformers import BertModel, DebertaModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ClassificationModelConcatenation(PreTrainedModel):
    config_class = GeneralConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.base_model_type == 'bert':
            self.transformer = BertModel.from_pretrained(config.base_model)
        elif config.base_model_type == 'deberta':
            self.transformer = DebertaModel.from_pretrained(config.base_model)
        else:
            raise ValueError("No valid base model type. " +
                             "Expected 'bert' or 'deberta', " +
                             f"got {config.base_model_type}")

        self.text_pooler = nn.Linear(config.text_dim, config.text_dim)
        self.text_pooler_fn = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(config.text_dim + config.num_features, config.num_labels)

        if config.freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, features, labels=None, **kwargs):
        seq_output = self.transformer(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      **kwargs)
        text_pooler_output = self.text_pooler_fn(
            self.text_pooler(
                seq_output.last_hidden_state[:, 0]
            )
        )
        concat_repr = torch.cat((text_pooler_output, features.float()),
                                dim=1)
        dropout_output = self.dropout(concat_repr)
        logits = self.linear(dropout_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.long())

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=seq_output.hidden_states,
                                        attentions=seq_output.attentions)


class ClassificationModelFeaturesPooling(ClassificationModelConcatenation):
    config_class = GeneralConfig

    def __init__(self, config):
        super().__init__(config)

        self.features_pooler = nn.Linear(config.num_features,
                                         config.num_features)
        self.features_pooler_fn = nn.GELU()

    def forward(self, input_ids, attention_mask, features, labels=None, **kwargs):
        seq_output = self.transformer(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      **kwargs)
        text_pooler_output = self.text_pooler_fn(
            self.text_pooler(
                seq_output.last_hidden_state[:, 0]
            )
        )
        features_pooler_output = self.features_pooler_fn(
            self.features_pooler(
                features.float()
            )
        )
        concat_repr = torch.cat((text_pooler_output, features_pooler_output),
                                dim=1)
        dropout_output = self.dropout(concat_repr)
        logits = self.linear(dropout_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.long())

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=seq_output.hidden_states,
                                        attentions=seq_output.attentions)


class ClassificationModelSharedRepresentation(ClassificationModelFeaturesPooling):
    config_class = SharedRepresentationConfig

    def __init__(self, config):
        super().__init__(config)

        self.shared_repr = nn.Linear(config.text_dim + config.num_features,
                                     config.hidden_dim)
        self.shared_repr_fn = nn.GELU()
        self.linear = nn.Linear(config.hidden_dim, config.num_labels)

    def forward(self, input_ids, attention_mask, features, labels=None, **kwargs):
        seq_output = self.transformer(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      **kwargs)
        text_pooler_output = self.text_pooler_fn(
            self.text_pooler(
                seq_output.last_hidden_state[:, 0]
            )
        )
        features_pooler_output = self.features_pooler_fn(
            self.features_pooler(
                features.float()
            )
        )
        concat_repr = torch.cat((text_pooler_output, features_pooler_output),
                                dim=1)
        shared_repr_output = self.shared_repr_fn(
            self.shared_repr(
                concat_repr
            )
        )
        dropout_output = self.dropout(shared_repr_output)
        logits = self.linear(dropout_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.long())

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=seq_output.hidden_states,
                                        attentions=seq_output.attentions)
