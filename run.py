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
    # mymodel = myBert.from_pretrained(checkpoint, config=myconfig).to(device)
    mymodel = torch.load('./train_model/role.bin').to(device)

    params = [
        {"params": mymodel.bert.parameters(), "lr": bert_learning_rate},
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
        # print(f"Epoch {epoch + 1}/{epoch_num}\n-------------------------------")
        # total_loss, batchs, batch_loss, total_average_loss = train(train_dataloader, mymodel, optimizer,
        #                                                            lr_scheduler, epoch + 1, device, total_loss,
        #                                                            batchs, batch_loss, total_average_loss)
        torch.save(mymodel.state_dict(), f'./train_model/{epoch + 1}model.bin')
        end_time = time.time()
        print('time', end_time - start_time)

        # for name, para in mymodel.named_parameters():
        #     if name == 'crf.transitions':
        #         print(para)

    #     for k, v in report_dic.items():
    #         v[0] = v[1] = v[2] = v[3] = v[4] = v[5] = 0
    #     test(dev_dataloader, mymodel, device)
    #     report()
    # draw(batchs, batch_loss, total_average_loss)
    print("Done!")


def report():
    sum_overlap_words = 0
    sum_pred_words = 0
    sum_label_words = 0

    sum_overlap_char = 0
    sum_pred_char = 0
    sum_label_char = 0

    sum_words_f1 = 0
    sum_char_f1 = 0
    for key, value in report_dic.items():
        sum_overlap_words += value[0]
        sum_pred_words += value[1]
        sum_label_words += value[2]

        sum_overlap_char += value[3]
        sum_pred_char += value[4]
        sum_label_char += value[5]

        if value[2] == 0:
            words_f1 = char_f1 = 1
        else:
            if value[1] == 0:
                words_precision = words_recall = char_precision = char_recall = 0
            else:
                words_precision = value[0] / value[1]
                words_recall = value[0] / value[2]
                char_precision = value[3] / value[4]
                char_recall = value[3] / value[5]
            if words_precision == 0 and words_recall == 0:
                words_f1 = 0
            else:
                words_f1 = 2 * words_precision * words_recall / (words_precision + words_recall)

            if char_precision == 0 and char_recall == 0:
                char_f1 = 0
            else:
                char_f1 = 2 * char_precision * char_recall / (char_precision + char_recall)

        sum_words_f1 += words_f1
        sum_char_f1 += char_f1
        print(key, ' 严格F1:', words_f1, ' 松弛F1:', char_f1)
    print('严格F1:', sum_words_f1/len(report_dic), '松弛F1:', sum_char_f1/len(report_dic))


if __name__ == '__main__':
    run()
