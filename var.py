import torch
from torch import nn
from transformers import AutoConfig, get_scheduler

from data_preprocess import train_dataloader
from model import myBert

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

bert_config = AutoConfig.from_pretrained('./bert-base-chinese')
bert_model = myBert.from_pretrained('./bert-base-chinese', config=bert_config).to(device)

learning_rate = 1e-5
epoch_num = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(bert_model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader),
)