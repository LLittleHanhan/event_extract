import json
import math

import matplotlib.pyplot as plt
import numpy as np

dev_path = './DuEE1.0/duee_dev.json'
test_path = './DuEE1.0/duee_test.json'
train_path = './DuEE1.0/duee_train.json'
label_path = './DuEE1.0/label.txt'
new_train_path = './DuEE1.0/new_duee_train.json'
schema_path = './DuEE1.0/duee_event_schema.json'
infor_path = './DuEE1.0/info.txt'


def draw(listx, listy):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(10, 40), dpi=200)
    color = ['red', 'peru', 'orchid', 'deepskyblue', 'green']
    plt.barh(listx, listy, color=color)
    plt.xticks(rotation=90, fontsize=8)
    for x, y, i in zip(listx, listy, range(len(listx))):
        plt.text(y + 50, i, y, verticalalignment='center', fontsize=20)
    plt.show()


# 测试分词
'''
{"text": "近日， “移动电影院V2.0”产品于北京正式 发布，恰逢移动电影院App首发一周年，引起了业内的高度关注。", "id": "8e8c78eec1e7eb9ebe942ebb25728d02", 
 "event_list": [
                {"event_type": "产品行为-发布", "trigger": "发布", "trigger_start_index": 22, 
                 "arguments": [
                                {"argument_start_index": 0, "role": "时间", "argument": "近日", "alias": []}, 
                                {"argument_start_index": 3, "role": "发布产品", "argument": " “移动电影院V2.0”产品", "alias": []}
                              ], 
                "class": "产品行为"}
               ]
}


'''
'''
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('./bert-base-chinese')
text = '近日, “移动电影院V2.0”产品于北京正式 发布，恰逢移动电影院App首发一周年，引起了业内的高度关注。'
encoding = tokenizer(text)

print(encoding.tokens())
print(encoding.word_ids())

batch_label = np.zeros(len(encoding['input_ids']), dtype=int)

token_start = encoding.char_to_token(23)
token_end = encoding.char_to_token(24)
print(token_start)
print(token_end)

batch_label[token_start] = 'B'
batch_label[token_start + 1:token_end + 1] = 'I'
'''

# 制作lable

with open(schema_path, 'r', encoding='utf-8') as s, open(label_path, 'w', encoding='utf-8') as l:
    data = ['0 O']
    idx = 1
    for line in s.readlines():
        json_data = json.loads(line)
        data.append(str(idx) + ' ' + 'B-' + json_data['event_type'])
        idx += 1
        data.append(str(idx) + ' ' + 'I-' + json_data['event_type'])
        idx += 1
    for d in data:
        l.write(d + '\n')


# 修复数据
'''
with open(train_path, 'r', encoding='utf-8') as f, open(new_train_path, 'w', encoding='utf-8') as nf:
    for line in f.readlines():
        json_data = json.loads(line)
        for event in json_data['event_list']:
            for argu in event['arguments']:
                if argu['argument'][0] == ' ':
                    argu['argument'] = str(argu['argument']).lstrip(' ')
                    argu['argument_start_index'] += 1
        nf.write(json.dumps(json_data, ensure_ascii=False) + '\n')
'''

# 统计事件
"""
{"text": "7月4日，由中铁十九局承建的青岛地铁1号线胜利桥站施工围挡处发生塌陷，造成一名施工人员死亡；而在此之前的5月27日，由中铁二十局施工的地铁4号线沙子口静沙区间施工段坍塌，5名被困工人全部遇难。", "id": "6d5b61216556ea335377a60923cf3ea5",
 "event_list": [{"event_type": "灾害/意外-坍/垮塌", "trigger": "塌陷", "trigger_start_index": 32,
                 "arguments": [{"argument_start_index": 0, "role": "时间", "argument": "7月4日", "alias": []},
                               {"argument_start_index": 14, "role": "坍塌主体", "argument": "青岛地铁1号线胜利桥站施工围挡处", "alias": []},
                               {"argument_start_index": 37, "role": "死亡人数", "argument": "一名", "alias": []}],
                 "class": "灾害/意外"},
                {"event_type": "人生-死亡", "trigger": "死亡", "trigger_start_index": 43,
                 "arguments": [{"argument_start_index": 0, "role": "时间", "argument": "7月4日", "alias": []},
                               {"argument_start_index": 39, "role": "死者", "argument": "施工人员", "alias": []},
                               {"argument_start_index": 14, "role": "地点", "argument": "青岛地铁1号线胜利桥站施工围挡处", "alias": []}],
                 "class": "人生"},
                {"event_type": "灾害/意外-坍/垮塌", "trigger": "坍塌", "trigger_start_index": 82,
                 "arguments": [{"argument_start_index": 52, "role": "时间", "argument": "5月27日", "alias": []},
                               {"argument_start_index": 67, "role": "坍塌主体", "argument": "地铁4号线沙子口静沙区间施工段", "alias": []},
                               {"argument_start_index": 85, "role": "死亡人数", "argument": "5名", "alias": []}],
                 "class": "灾害/意外"},
                {"event_type": "人生-死亡", "trigger": "遇难", "trigger_start_index": 93,
                 "arguments": [{"argument_start_index": 52, "role": "时间", "argument": "5月27日", "alias": []},
                               {"argument_start_index": 87, "role": "死者", "argument": "被困工人", "alias": []},
                               {"argument_start_index": 67, "role": "地点", "argument": "地铁4号线沙子口静沙区间施工段", "alias": []}],
                 "class": "人生"}]}
"""
'''
event_type_list_all = []
event_type_list = []
with open(schema_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_data = json.loads(line)
        event_type_list_all.append(str(json_data['event_type']))
        event_type_list.append(str(json_data['event_type']).split('-')[1])
event_type_num = [0 for i in range(len(event_type_list_all))]  # 总各类事件数量
event_type_label2id = {}
event_type_id2label = {}

for idx, event_type in enumerate(event_type_list_all):
    event_type_label2id[event_type] = idx
    event_type_id2label[idx] = event_type

text_len = []  # 文本长度
every_text_event_num = []  # 每一文本的事件数
with open(train_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_data = json.loads(line)
        text_len.append(len(json_data['text']))
        every_text_event_num.append(len(json_data['event_list']))
        for event in json_data['event_list']:
            event_type_num[event_type_label2id[event['event_type']]] += 1

every_text_event_num_dic = {}
for event_num in every_text_event_num:
    if event_num in every_text_event_num_dic:
        every_text_event_num_dic[event_num] += 1
    else:
        every_text_event_num_dic[event_num] = 1

len_dic={100:0,200:0,300:0,400:0,500:0}
for len in text_len:
    if 0 < len <= 100:
        len_dic[100] += 1
    elif 100<len<=200:
        len_dic[200] += 1
    elif 200<len<=300:
        len_dic[300] += 1
    elif 300<len<=400:
        len_dic[400] += 1
    else:
        len_dic[500] +=1
print(len_dic)

with open(infor_path, 'w', encoding='utf-8') as f:
    for event_type, num in zip(event_type_list, event_type_num):
        f.write(event_type + ' ' + str(num) + '\n')
draw(event_type_list, event_type_num)
'''
