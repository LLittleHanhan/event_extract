import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoConfig

from var import device, dev_path, train_path, checkpoint
from data_preprocess import myDataSet, collote_fn
from train import train, test, draw
from model import myBert

train_batch_size = 12
dev_batch_size = 64

dev_data = myDataSet(dev_path)
dev_dataloader = DataLoader(dev_data, batch_size=dev_batch_size, shuffle=True, collate_fn=collote_fn)
train_data = myDataSet(train_path)
train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collote_fn)

myconfig = AutoConfig.from_pretrained(checkpoint)
mymodel = myBert.from_pretrained(checkpoint, config=myconfig).to(device)

bert_learning_rate = 2e-5
CRF_learning_rate = 2e-4
epoch_num = 10
loss_fn = nn.CrossEntropyLoss()
params = [
    {"params": mymodel.bert.parameters(), "lr": bert_learning_rate},
    {"params": mymodel.classifier.parameters(), "lr": CRF_learning_rate},
    {"params": mymodel.crf.parameters(), "lr": CRF_learning_rate},
]
optimizer = torch.optim.AdamW(params)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0.1 * epoch_num * len(train_dataloader),
    num_training_steps=epoch_num * len(train_dataloader),
)

total_loss = 0.
batchs = []
batch_loss = []
total_average_loss = []

for epoch in range(epoch_num):
    start_time = time.time()
    print(f"Epoch {epoch + 1}/{epoch_num}\n-------------------------------")
    total_loss, batchs, batch_loss, total_average_loss = train(train_dataloader, mymodel, loss_fn, optimizer,
                                                               lr_scheduler, epoch + 1, device, total_loss,
                                                               batchs, batch_loss, total_average_loss)
    test(dev_dataloader, mymodel, device)
    end_time = time.time()
    print('time', end_time - start_time)
    torch.save(mymodel, f'./train_model/{epoch + 1}triggerModel.bin')
draw(batchs, batch_loss, total_average_loss)
print("Done!")
