import torch
import time
from torch.utils.data import DataLoader
from transformers import get_scheduler

from var import device, dev_path, train_path, dev_batch_size, train_batch_size, epoch_num, \
    report_dic, bert_learning_rate, CRF_learning_rate
from data_preprocess import myDataSet, collote_fn
from train import train, test, draw
from model import myModel


def run():
    dev_data = myDataSet(dev_path)
    dev_dataloader = DataLoader(dev_data, batch_size=dev_batch_size, shuffle=True, collate_fn=collote_fn)
    train_data = myDataSet(train_path)
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collote_fn)

    mymodel = myModel().to(device)
    # mymodel = torch.load('./train_model/2.bin').to(device)

    params = [
        {"params": mymodel.ernie.parameters(), "lr": bert_learning_rate},
        {"params": mymodel.trigger_embedding.parameters(), "lr": CRF_learning_rate},
        # {"params": mymodel.mid_linear.parameters(), "lr": CRF_learning_rate},
        {"params": mymodel.lay_norm.parameters(), "lr": CRF_learning_rate},
        {"params": mymodel.classifier.parameters(), "lr": CRF_learning_rate},
        {"params": mymodel.crf.parameters(), "lr": CRF_learning_rate},
    ]
    optimizer = torch.optim.AdamW(params)
    for lr in optimizer.state_dict()["param_groups"]:
        print(lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=epoch_num * len(train_dataloader) * 0.1,
        num_training_steps=epoch_num * len(train_dataloader),
    )

    total_loss = 0.
    batchs = []
    batch_loss = []
    total_average_loss = []
    for epoch in range(epoch_num):

        start_time = time.time()
        print(f"Epoch {epoch + 1}/{epoch_num}\n-------------------------------")
        total_loss, batchs, batch_loss, total_average_loss = train(train_dataloader, mymodel, optimizer,
                                                                   lr_scheduler, epoch + 1, device, total_loss,
                                                                   batchs, batch_loss, total_average_loss)
        torch.save(mymodel, f'./train_model/{epoch + 1}model.bin')
        end_time = time.time()
        print('time', end_time - start_time)

        # for name, para in mymodel.named_parameters():
        #     if name == 'crf.transitions':
        #         print(para)

        for k, v in report_dic.items():
            v[0] = v[1] = v[2] = v[3] = v[4] = 0
        test(dev_dataloader, mymodel, device)
        report()
    draw(batchs, batch_loss, total_average_loss)
    print("Done!")


def report():
    sum_r = 0
    sum_w = 0
    no_answer = 0
    surplus = 0
    c_sum_r = 0
    for k, v in report_dic.items():
        sum_r += v[0]
        sum_w += v[1]
        no_answer += v[2]
        surplus += v[3]
        c_sum_r += v[4]
        if v[0] + v[1] == 0:
            acc = 1.
            c_acc = 1.
        else:
            acc = float(v[0]) / (v[0] + v[1])
            c_acc = float(v[4]) / (v[0] + v[1])
        # print(k, ' 正确:', v[0], ' 错误:', v[1], '空:', v[2], '多:', v[3], '模糊正确:', v[4], ' 正确率:', acc, '模糊正确率:', c_acc)
    print(sum_r, sum_w, no_answer, surplus, c_sum_r)
    print('总正确率:', float(sum_r) / (sum_r + sum_w))
    print('模糊正确率:', float(c_sum_r) / (sum_r + sum_w))


if __name__ == '__main__':
    run()
