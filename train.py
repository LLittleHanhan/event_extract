import re
import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from var import id2label, test_result_path, schema_path, checkpoint, report_dic


def train(dataloader, model, optimizer, lr_scheduler, epoch, device, total_loss, batchs, batch_loss,
          total_average_loss):
    model.train()
    finish_batch_num = (epoch - 1) * len(dataloader)

    for batch, (X, y, z) in enumerate(dataloader, start=1):
        X = X.to(device)
        y = y.to(device)
        z = z.to(device)
        _, loss = model(X, z, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        if batch % 2 == 0:
            total_batch = finish_batch_num + batch
            print('train:batch:', batch, '/', len(dataloader), '\t\t\t', 'loss:', total_loss / total_batch)
            batchs.append(total_batch)
            batch_loss.append(loss.item())
            total_average_loss.append(total_loss / total_batch)
    return total_loss, batchs, batch_loss, total_average_loss


def test(dataloader, model, device):
    model.eval()
    with torch.no_grad():
        for idx, (X, y, z) in enumerate(dataloader, start=1):
            X, y, z = X.to(device), y.to(device), z.to(device)

            _, preds = model(X, z)

            labels = y.cpu().numpy().tolist()

            end = X['attention_mask'].sum(dim=1)
            start = X['attention_mask'].sum(dim=1) - X['token_type_ids'].sum(dim=1)
            targets = [labels[idx][start[idx].item():end[idx].item()] for idx in range(len(labels))]

            analyze(preds, targets, X)

            if idx % 100 == 0:
                print('test:', idx, '/', len(dataloader))


def draw(batchs, batch_loss, total_average_loss):
    plt.title = 'loss'
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.plot(batchs, batch_loss, color='red', label='batch_loss')
    plt.plot(batchs, total_average_loss, color='green', label='total_average_loss')
    plt.legend(loc="best")
    plt.show()


def analyze(preds, true_labels, X):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    for (pred, label, input_ids) in zip(preds, true_labels, X['input_ids']):
        seq = str(tokenizer.decode(input_ids, skip_special_tokens=True)).replace(' ', '')
        seq = re.split('触发词为|的事件|中角色|是什么?', seq)
        event_type = seq[2]
        event_trigger = seq[1]
        role = seq[3]
        if pred == label:
            report_dic[event_type + '-' + role][0] += 1

        else:
            report_dic[event_type + '-' + role][1] += 1
