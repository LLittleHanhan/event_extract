import torch
# 文件路径
dev_path = './DuEE1.0/duee_dev.json'
test_path = './DuEE1.0/duee_test.json'
train_path = './DuEE1.0/new_duee_train.json'
label_path = './DuEE1.0/label.txt'
test_result_path = './test_result.txt'
# id2label
label2id = {}
id2label = {}
with open(label_path) as f:
    for line in f.readlines():
        id, label = line.strip().split(' ')
        label2id[label] = int(id)
        id2label[int(id)] = label

#
checkpoint = './bert-base-chinese'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')



