import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import matplotlib.pyplot as plt
from sklearn import metrics

from var import test_result_path


def train(dataloader, model, optimizer, lr_scheduler, epoch, device, total_loss, batchs, batch_loss,
          total_average_loss):
    model.train()
    finish_batch_num = (epoch - 1) * len(dataloader)

    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        _, loss = model(X, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        if batch % 100 == 0:
            total_batch = finish_batch_num + batch
            print('train:batch:', batch, '/', len(dataloader), '\t\t\t', 'loss:', total_loss / total_batch)
            batchs.append(total_batch)
            batch_loss.append(loss.item())
            total_average_loss.append(total_loss / total_batch)
    return total_loss, batchs, batch_loss, total_average_loss


def test(dataloader, model, device):
    true_labels, true_predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader, start=1):
            X, y = X.to(device), y.to(device)
            pred, _ = model(X)
            pred = pred.argmax(dim=-1).cpu().numpy().tolist()
            label = y.cpu().numpy().tolist()

            true_labels += label
            true_predictions += pred
    report = metrics.classification_report(true_labels, true_predictions,
                                           target_names=['finance', 'stocks', 'education', 'science', 'society',
                                                         'politics', 'sports', 'game', 'entertainment'], digits=2)
    confusion = metrics.confusion_matrix(true_labels, true_predictions)
    print(report)
    print(confusion)


def draw(batchs, batch_loss, total_average_loss):
    plt.title = 'loss'
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.plot(batchs, batch_loss, color='red', label='batch_loss')
    plt.plot(batchs, total_average_loss, color='green', label='total_average_loss')
    plt.legend(loc="best")
    plt.show()
