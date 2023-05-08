import torch
import time
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoConfig

from var import device, dev_path, train_path, checkpoint, dev_batch_size, train_batch_size, epoch_num, \
    report_dic, bert_learning_rate, CRF_learning_rate
from data_preprocess import myDataSet, collote_fn
from train import train, test, draw
from model import myBert


def run():
    dev_data = myDataSet(dev_path)
    dev_dataloader = DataLoader(dev_data, batch_size=dev_batch_size, shuffle=True, collate_fn=collote_fn)
    train_data = myDataSet(train_path)
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collote_fn)

    myconfig = AutoConfig.from_pretrained(checkpoint)
    mymodel = myBert.from_pretrained(checkpoint, config=myconfig).to(device)
    # mymodel = torch.load('./train_model/1model.bin').to(device)

    params = [
        {"params": mymodel.bert.parameters(), "lr": bert_learning_rate},
        {"params": mymodel.trigger_embedding.parameters(), "lr": CRF_learning_rate},
        {"params": mymodel.lay_norm.parameters(), "lr": CRF_learning_rate},
        {"params": mymodel.mid_linear.parameters(), "lr": CRF_learning_rate},
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
            v[0] = v[1] = 0
        test(dev_dataloader, mymodel, device)
        sum_r = 0
        sum_w = 0
        for k, v in report_dic.items():
            sum_r += v[0]
            sum_w += v[1]
            if v[0] + v[1] == 0:
                acc = 1.
            else:
                acc = float(v[0]) / (v[0] + v[1])
            print(k, ' 正确:', v[0], ' 错误:', v[1], ' 正确率:', acc)
        print(sum_r, sum_w)
        print('总正确率:', float(sum_r) / (sum_r + sum_w))
    # draw(batchs, batch_loss, total_average_loss)
    print("Done!")


if __name__ == '__main__':
    run()
