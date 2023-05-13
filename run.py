import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoConfig

from var import device, dev_path, train_path, checkpoint
from data_preprocess import myDataSet, collote_fn
from train import train, test, draw
from model import myBert

train_batch_size = 128
dev_batch_size = 256

dev_data = myDataSet(dev_path)
dev_dataloader = DataLoader(dev_data, batch_size=dev_batch_size, shuffle=True, collate_fn=collote_fn)
train_data = myDataSet(train_path)
train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collote_fn)

myconfig = AutoConfig.from_pretrained(checkpoint)
# mymodel = myBert.from_pretrained(checkpoint, config=myconfig).to(device)
mymodel = torch.load('./train_model/class.bin').to(device)

bert_learning_rate = 2e-5
other_learning_rate = 2e-4
epoch_num = 3

params = [
    {"params": mymodel.bert.parameters(), "lr": bert_learning_rate},
    {"params": mymodel.classifier.parameters(), "lr": other_learning_rate},
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
    # total_loss, batchs, batch_loss, total_average_loss = train(train_dataloader, mymodel, optimizer,
    #                                                            lr_scheduler, epoch + 1, device, total_loss,
    #                                                            batchs, batch_loss, total_average_loss)
    torch.save(mymodel.state_dict(), f'./train_model/class.bin')

    # test(dev_dataloader, mymodel, device)
    end_time = time.time()
    print('time', end_time - start_time)

draw(batchs, batch_loss, total_average_loss)
print("Done!")
