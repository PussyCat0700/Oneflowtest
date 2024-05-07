from copy import deepcopy
from omegaconf import DictConfig
import transformers
import libai
class BertForClassificationLiBai(libai.models.BertForClassification):
    def __init__(self, cfg):
        config = DictConfig(cfg)
        super().__init__(config)
    
    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        outputs = self.bert(input_ids, attention_mask, tokentype_ids)
        pooled_output = outputs[0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
class BertForClassificationTorch(transformers.BertForTokenClassification):
    def __init__(self, cfg):
        cfg = deepcopy(cfg)
        config = transformers.T5Config(
            type_vocab_size=cfg.pop('num_tokentypes'),
            layer_norm_eps=cfg.pop('layernorm_eps'),
            hidden_act='gelu',  # Same with BertLMPredictionHead in OF.
            **cfg
        )
        super().__init__(config)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits