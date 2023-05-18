import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig

from model import myBert
from var import device, id2label

label_path = './DuEE1.0/label.txt'
label2id = {}
id2label = {}
with open(label_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        id, label = line.strip().split(' ')
        label2id[label] = int(id)
        id2label[int(id)] = label

sentence = '如何看待“恒大国脚因伤退出国足后迅速在联赛复出”这个热议话题'

tokenizer = AutoTokenizer.from_pretrained('./chinese-roberta-wwm-ext')
config = AutoConfig.from_pretrained('./chinese-roberta-wwm-ext')
model = myBert(config)
model.load_state_dict(torch.load('./train_model/1triggerModel.bin'))
model.eval()
with torch.no_grad():
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt", return_offsets_mapping=True, max_length=512)
    offsets = inputs.pop('offset_mapping').squeeze(0)
    print(offsets[0])
    start,end = offsets[0]
    print(start,end)
    pred = model(inputs)[1][0][1:-1]
    print(pred)
    idx = 0
    trigger = 0
    trigger_start = trigger_end = []
    while idx < len(pred):
        if pred[idx] % 2 == 1:
            print('yes')
            trigger = pred[idx]
            start, end = offsets[idx]
            trigger_start.append(start)
        while idx+1 < len(pred) and pred[idx + 1] == trigger + 1:
            idx += 1
        _, end = offsets[idx]
        trigger_end.append(end)
        idx += 1
    print(trigger_start)
    print(trigger_end)
