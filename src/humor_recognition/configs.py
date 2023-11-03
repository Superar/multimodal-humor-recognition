from transformers import PretrainedConfig


class GeneralConfig(PretrainedConfig):
    def __init__(self,
                 base_model='neuralmind/bert-base-portuguese-cased',
                 base_model_type='bert',
                 num_labels=2,
                 num_features=27,
                 freeze_base=False,
                 **kwargs):
        self.base_model = base_model
        self.base_model_type = base_model_type
        self.text_dim = 768 if base_model_type == 'bert' else 1536
        self.num_labels = num_labels
        self.num_features = num_features
        self.freeze_base = freeze_base
        super().__init__(**kwargs)


class SharedRepresentationConfig(GeneralConfig):
    def __init__(self, hidden_dim=786, **kwargs):
        self.hidden_dim = hidden_dim
        super().__init__(**kwargs)
