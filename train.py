import torch
from data_preprocess import id2label

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss, device):
    finish_batch_num = (epoch - 1) * len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        # print(pred.size())
        # print(y.size())

        loss = loss_fn(pred.permute(0, 2, 1), y)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        if batch%5 == 0:
            print('batch:', batch, '/', len(dataloader), '\t\t\t', 'loss:', total_loss / (finish_batch_num + batch))
    return total_loss


def test_loop(dataloader, model, device):
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
