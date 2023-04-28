import re
import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from var import id2label, test_result_path, schema_path, checkpoint, report_dic


def train(dataloader, model, optimizer, lr_scheduler, epoch, device, total_loss, batchs, batch_loss,
          total_average_loss, isCRF=True):
    model.train()
    finish_batch_num = (epoch - 1) * len(dataloader)

    for batch, (X, y, z) in enumerate(dataloader, start=1):
        X = X.to(device)
        if isCRF:
            y = y.to(device)
            _, loss = model(X, y, isCRF)
        else:
            z = z.to(device)
            _, loss = model(X, z, isCRF)

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


def test(dataloader, model, device, isCRF=True):
    model.eval()
    with torch.no_grad():
        for idx, (X, y, z) in enumerate(dataloader, start=1):
            X, z = X.to(device), z.to(device)

            _, preds = model(X, isCRF=isCRF)

            # z用作标签和过滤，只看第二句话评估
            labels = z.cpu().numpy().tolist()

            predictions = [
                [p for (p, t) in zip(one_p, one_t) if t != -100]
                for one_p, one_t in zip(preds, labels)
            ]
            targets = [[t for t in one_t if t != -100] for one_t in labels]

            report(predictions, targets, X)

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


def report(preds, true_labels, X):
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
