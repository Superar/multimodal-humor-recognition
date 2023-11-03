from collections import OrderedDict
from typing import Dict, List, Union
from transformers.pipelines import TextClassificationPipeline
from transformers.pipelines.base import GenericTensor

SUPPORTED_MODELS = OrderedDict([
    ('features-pooling', 'ClassificationModelFeaturesPooling'),
    ('shared-representation', 'ClassificationModelSharedRepresentation'),
    ('concatenation', 'ClassificationModelConcatenation')
])


class MultimodalPipeline(TextClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_model_type(self, supported_models: List[str] | dict):
        return super().check_model_type(SUPPORTED_MODELS)

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        num_inputs = inputs['input_ids'].shape[0]
        num_features = inputs['features'].shape[0]
        return {'input_ids': inputs['input_ids'].view(-1, num_inputs),
                'attention_mask': inputs['attention_mask'].view(-1, num_inputs),
                'features': inputs['features'].view(-1, num_features)}