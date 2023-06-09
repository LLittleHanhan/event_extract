import re
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from var import checkpoint, report_dic


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
        if batch % 5 == 0:
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

            if idx % 500 == 0:
                print('test:', idx, '/', len(dataloader))


def draw(batchs, batch_loss, total_average_loss):
    plt.title = 'loss'
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.plot(batchs, batch_loss, color='red', label='batch_loss')
    plt.plot(batchs, total_average_loss, color='green', label='total_average_loss')
    plt.legend(loc="best")
    plt.show()


# 预测和实际论元重叠，预测论元数，实际论元数
# 预测和实际重叠字符数，预测字符数，实际字符数

def analyze(preds, true_labels, X):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    for (pred, label, input_ids) in zip(preds, true_labels, X['input_ids']):
        seq = str(tokenizer.decode(input_ids, skip_special_tokens=True)).replace(' ', '')
        seq = re.split('触发词为|的事件|中角色|是什么？', seq)
        event_type = seq[2]
        event_trigger = seq[1]
        role = seq[3]

        pred_start = []
        pred_end = []
        idx = 0
        while idx < len(pred):
            if pred[idx] == 1:
                pred_start.append(idx)
                while idx + 1 < len(pred) and pred[idx + 1] == 2:
                    idx += 1
                pred_end.append(idx)
            idx += 1

        label_start = []
        label_end = []
        idx = 0
        while idx < len(label):
            if label[idx] == 1:
                label_start.append(idx)
                while idx + 1 < len(label) and label[idx + 1] == 2:
                    idx += 1
                label_end.append(idx)
            idx += 1

        words_overlap = 0
        for idx, s in enumerate(pred_start):
            if s in label_start and pred_end[idx] == label_end[label_start.index(s)]:
                words_overlap += 1
        report_dic[event_type + '-' + role][0] += words_overlap
        report_dic[event_type + '-' + role][1] += len(pred_start)
        report_dic[event_type + '-' + role][2] += len(label_start)

        char_overlap = 0
        for idx, c in enumerate(label):
            if c != 0 and pred[idx] != 0:
                char_overlap += 1
        report_dic[event_type + '-' + role][3] += char_overlap
        report_dic[event_type + '-' + role][4] += sum(pred_end) - sum(pred_start) + len(pred_start)
        report_dic[event_type + '-' + role][5] += sum(label_end) - sum(label_start) + len(label_start)

        # print(pred)
        # print(label)
        # print(pred_start)
        # print(pred_end)
        # print(label_start)
        # print(label_end)
        #
        # print(words_overlap)
        # print(char_overlap)
        # print(sum(pred_end) - sum(pred_start) + len(pred_start))
        # print(sum(label_end) - sum(label_start) + len(label_start))
