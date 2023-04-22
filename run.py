import torch
from torch import nn
from transformers import get_scheduler

from data_preprocess import train_dataloader, dev_dataloader
from model import model
from train import train_loop, test_loop

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

learning_rate = 1e-5
epoch_num = 1
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader),
)
total_loss = 0.

for t in range(epoch_num):
    print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t + 1, total_loss, device)
    # test_loop(dev_dataloader, model, device)
print("Done!")
