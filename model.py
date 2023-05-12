import torch
from torch import nn
from transformers import ErnieModel, AutoConfig
from crf import CRF
from var import id2label, checkpoint


class myModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(checkpoint)
        self.ernie = ErnieModel.from_pretrained(checkpoint, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.trigger_embedding = nn.Embedding(num_embeddings=5, embedding_dim=16)
        # 16+768
        self.lay_norm = nn.LayerNorm(784, eps=config.layer_norm_eps)



        self.classifier = nn.Linear(784, len(id2label))
        self.crf = CRF(len(id2label), batch_first=True)

    def forward(self, x, trigger_position, label=None):
        # with torch.no_grad():
        ernie_output = self.ernie(**x)
        sequence_output = ernie_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)


        trigger_position_feature = self.trigger_embedding(trigger_position)

        sequence_output = torch.cat([sequence_output, trigger_position_feature], dim=-1)
        sequence_output = self.lay_norm(sequence_output)

        logits = self.classifier(sequence_output)

        # crf修正

        shift = x['attention_mask'].sum(dim=1) - x['token_type_ids'].sum(dim=1)

        re_token_type = torch.stack(
            [torch.roll(x['token_type_ids'][idx], -shift[idx].item(), 0) for idx in range(shift.size(dim=0))],
            dim=0)
        re_logits = torch.stack(
            [torch.roll(logits[idx], -shift[idx].item(), 0) for idx in range(shift.size(dim=0))],
            dim=0)

        # 计算loss
        if label is not None:

            label = torch.stack(
                [torch.roll(label[idx], -shift[idx].item(), 0) for idx in range(shift.size(dim=0))],
                dim=0)

            loss = -self.crf(emissions=re_logits, tags=label, mask=re_token_type, reduction="token_mean")
        else:
            # 这里的loss用作pred
            loss = self.crf.decode(emissions=re_logits, mask=re_token_type)
        return logits, loss


if __name__ == '__main__':

    model = myModel()
    print(model)
    for idx, (name, para) in enumerate(model.named_parameters()):
        print(idx + 1, name, para.shape, para.numel())
    total = sum(p.numel() for p in model.parameters())
    print(total)
