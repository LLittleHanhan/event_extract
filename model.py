import torch

from data_preprocess import id2label
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig


class myBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, len(id2label))
        self.post_init()

    def forward(self, x):
        with torch.no_grad():
            bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


if __name__ == '__main__':
    config = AutoConfig.from_pretrained('./bert-base-chinese')
    model = myBert.from_pretrained('./bert-base-chinese', config=config)
    for name, para in model.named_parameters():
        print(name, para.shape, para.numel())
    total = sum(p.numel() for p in model.parameters())
    print(total)
