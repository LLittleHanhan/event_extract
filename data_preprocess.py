'''
斐济洪灾持续数日至少6人死亡	5
'''

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from var import checkpoint


class myDataSet(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                title, label = l.strip().split('\t')
                dic = {'title': title, 'label': label}
                self.data.append(dic)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collote_fn(batch_samples):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    batch_text, batch_label = [], []
    for sample in batch_samples:
        batch_text.append(sample['title'])
        batch_label.append(int(sample['label']))

    batch_inputs = tokenizer(
        batch_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    batch_label =  torch.LongTensor(batch_label)
    return batch_inputs, batch_label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from var import dev_path, train_path

    dev_data = myDataSet(dev_path)
    dev_dataloader = DataLoader(dev_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    train_data = myDataSet(train_path)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)

    it = (iter(train_dataloader))
    while True:
        batch_X, batch_y = next(it)
        # print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
        # print('batch_y shape:', batch_y.shape)
        print(batch_X)
        print(batch_y)


