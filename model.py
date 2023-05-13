from torch import nn
from transformers import BertPreTrainedModel, BertModel

from var import checkpoint


class myBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 9)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, x, label=None):
        # with torch.no_grad():
        bert_output = self.bert(**x)
        sequence_output = bert_output.pooler_output
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)
        return logits, loss


if __name__ == '__main__':
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint)
    model = myBert.from_pretrained(checkpoint, config=config)
    print(model)
    for name, para in model.named_parameters():
        print(name, para.shape, para.numel())
    total = sum(p.numel() for p in model.parameters())
    print(total)
