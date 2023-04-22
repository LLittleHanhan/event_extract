import torch
from torch import nn

from model import myBert
from data_preprocess import train_dataloader, dev_dataloader, id2label

from transformers import AutoConfig
from transformers import AdamW, get_scheduler

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    finish_batch_num = (epoch - 1) * len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred.permute(0, 2, 1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        print('batch:', batch, '/', len(dataloader), '\t\t\t', 'loss:', total_loss / (finish_batch_num + batch))
    return total_loss


def test_loop(dataloader, model):
    true_labels, true_predictions = [], []
    model.eval()
    with torch.no_grad():
        idx = 1
        for X, y in dataloader:
            if (idx - 1) % 10 == 0:
                print(idx, '/', len(dataloader))
            idx += 1

            X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions = pred.argmax(dim=-1).cpu().numpy().tolist()
            labels = y.cpu().numpy().tolist()

            true_labels += [[id2label[int(l)] for l in label if l != -100] for label in labels]
            true_predictions += [
                [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
config = AutoConfig.from_pretrained('./bert-base-chinese')
model = myBert.from_pretrained('./bert-base-chinese', config=config).to(device)

learning_rate = 1e-5
epoch_num = 1
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader),
)
total_loss = 0.

for t in range(epoch_num):
    print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t + 1, total_loss)
    test_loop(dev_dataloader, model)
print("Done!")
