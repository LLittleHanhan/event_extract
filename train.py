import torch
from data_preprocess import id2label

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

import matplotlib.pyplot as plt


def train(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch_num, device):
    batchs = []
    batch_loss = []
    total_average_loss = []

    total_loss = 0.

    model.train()
    for epoch in range(epoch_num):

        print(f"Epoch {epoch + 1}/{epoch_num}\n-------------------------------")
        finish_batch_num = epoch * len(dataloader)

        for batch, (X, y) in enumerate(dataloader, start=1):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # print(pred.size())
            # print(y.size())
            loss = loss_fn(pred.permute(0, 2, 1), y)
            optimizer.zero_grad()
            loss.backward()
            # for name, para in model.named_parameters():
            #     if name == 'bert.encoder.layer.0.attention.self.query.weight':
            #         print(para)
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            if batch % 10 == 0:
                total_batch = finish_batch_num + batch
                print('batch:', batch, '/', len(dataloader), '\t\t\t', 'loss:', total_loss / total_batch)
                batchs.append(total_batch)
                batch_loss.append(loss.item())
                total_average_loss.append(total_loss / total_batch)
    draw(batchs, batch_loss, total_average_loss)
    return


def test(dataloader, model, device):
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


def draw(batch, batch_loss, total_average_loss):
    plt.title = 'loss'
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.plot(batch, batch_loss, color='red', label='batch_loss')
    plt.plot(batch, total_average_loss, color='green', label='total_average_loss')
    plt.legend(loc="best")
    plt.show()
