import json
import torch

# 文件路径
dev_path = './DuEE1.0/duee_dev.json'
test_path = './DuEE1.0/duee_test.json'
train_path = './DuEE1.0/new_duee_train.json'
label_path = './DuEE1.0/label.txt'
test_result_path = './test_result.txt'
schema_path = './DuEE1.0/duee_event_schema.json'
infor_path = './DuEE1.0/info.txt'

# id2label

# with open(schema_path, 'r', encoding='utf-8') as s, open(label_path, 'w', encoding='utf-8') as l:
#     data = ['0 O']
#     idx = 1
#     for line in s.readlines():
#         json_data = json.loads(line)
#         data.append(str(idx) + ' ' + 'B-' + json_data['event_type'])
#         idx += 1
#         data.append(str(idx) + ' ' + 'I-' + json_data['event_type'])
#         idx += 1
#         # for role in json_data['role_list']:
#         #     data.append(str(idx) + ' ' + 'B-' + json_data['event_type'] + '-' + role['role'])
#         #     idx += 1
#         #     data.append(str(idx) + ' ' + 'I-' + json_data['event_type'] + '-' + role['role'])
#         #     idx +=1
#     for d in data:
#         l.write(d + '\n')
label2id = {}
id2label = {}
with open(label_path, 'r',encoding='utf-8') as f:
    for line in f.readlines():
        id, label = line.strip().split(' ')
        label2id[label] = int(id)
        id2label[int(id)] = label

#
report_dic = {}
with open(schema_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_data = json.loads(line)
        event_type = str(json_data['event_type']).split('-')[1]
        for role in json_data['role_list']:
            report_dic[event_type + '-' + role['role']] = [0,0]

#
checkpoint = './chinese-roberta-wwm-ext'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
train_batch_size = 8
dev_batch_size = 64
CRF_learning_rate = 2e-4
bert_learning_rate = 2e-5
epoch_num = 5
