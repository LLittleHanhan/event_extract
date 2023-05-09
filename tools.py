import json

from matplotlib import pyplot as plt

from data_preprocess import myDataSet

dev_path = './DuEE1.0/duee_dev.json'
test_path = './DuEE1.0/duee_test.json'
train_path = './DuEE1.0/duee_train.json'
label_path = './DuEE1.0/label.txt'
new_train_path = './DuEE1.0/new_duee_train.json'
schema_path = './DuEE1.0/duee_event_schema.json'
infor_path = './DuEE1.0/info.txt'

# # 修复数据
#
# with open(train_path, 'r', encoding='utf-8') as f, open(new_train_path, 'w', encoding='utf-8') as nf:
#     for line in f.readlines():
#         json_data = json.loads(line)
#         json_data['text'] = str(json_data['text']).strip()
#         for event in json_data['event_list']:
#             for argu in event['arguments']:
#                 if argu['argument'][0] == ' ':
#                     argu['argument'] = str(argu['argument']).lstrip(' ')
#                     argu['argument_start_index'] += 1
#         nf.write(json.dumps(json_data, ensure_ascii=False) + '\n')


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
def draw(listx, listy):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(20, 80), dpi=100)
    color = ['red', 'peru', 'orchid', 'deepskyblue', 'green']
    plt.barh(listx, listy, color=color)
    plt.xticks(rotation=90, fontsize=1)
    for x, y, i in zip(listx, listy, range(len(listx))):
        plt.text(y + 50, i, y, verticalalignment='center', fontsize=10)
    plt.show()


event_type_list = []  # 事件类型
event_role_list = []  # 事件类型
text_len = []  # 文本长度

text_event_num = []  # 每一文本的事件数
text_event_num_dic = {}

event_type_label2id = {}
event_type_id2label = {}
event_role_label2id = {}
event_role_id2label = {}

# 事件类型，角色数量统计
with open(schema_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_data = json.loads(line)
        event_type_list.append(str(json_data['event_type']))
        for role in json_data['role_list']:
            event_role_list.append(str(json_data['event_type']) +'-' + role['role'])

event_type_num = [0 for i in range(len(event_type_list))]  # 总各类事件数量
event_role_num = [0 for i in range(len(event_role_list))]  # 总各类事件角色数量

for idx, event_type in enumerate(event_type_list):
    event_type_label2id[event_type] = idx
    event_type_id2label[idx] = event_type
for idx, event_role in enumerate(event_role_list):
    event_role_label2id[event_role] = idx
    event_role_id2label[idx] = event_role

with open(train_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_data = json.loads(line)

        text_len.append(len(json_data['text']))
        text_event_num.append(len(json_data['event_list']))

        for event in json_data['event_list']:
            event_type_num[event_type_label2id[event['event_type']]] += 1
            for argu in event['arguments']:
                event_role_num[event_role_label2id[event['event_type']+'-'+argu['role']]] += 1
draw(event_type_list, event_type_num)
draw(event_role_list, event_role_num)

# 单文本事件类型数量统计
for event_num in text_event_num:
    if event_num in text_event_num_dic:
        text_event_num_dic[event_num] += 1
    else:
        text_event_num_dic[event_num] = 1
print(sorted(text_event_num_dic.items(), key=lambda kv: kv[1]))

# 文本长度统计
len_dic = {'0~100': 0, '100~200': 0, '200~300': 0, '300~400': 0, '400~': 0}
for len in text_len:
    if 0 < len <= 100:
        len_dic['0~100'] += 1
    elif 100 < len <= 200:
        len_dic['100~200'] += 1
    elif 200 < len <= 300:
        len_dic['200~300'] += 1
    elif 300 < len <= 400:
        len_dic['300~400'] += 1
    else:
        len_dic['400~'] += 1
print(len_dic)

'''

'''
'text': '6月7日报道，IBM将裁员超过1000人。IBM周四确认，将裁减一千多人。据知情人士称，此次裁员将影响到约1700名员工，约占IBM全球逾34万员工中的0.5%。IBM股价今年累计上涨16%，但该公司4月发布的财报显示，一季度营收下降5%，低于市场预期。', 
'event_type': '组织关系-裁员', 
'trigger': '裁员', 
'trigger_start': 11, 
'trigger_end': 12, 
'role': '裁员方', 
'argu': ['IBM'], 
'argu_start': [7], 
'argu_end': [9]
}
'''
dataset = myDataSet(train_path)
for data in dataset:
    event_type = str(data['event_type']).replace('/', '&&')
    role = str(data['role']).replace('/', '&&')
    path = './data_analyse/' + event_type + '----' + role + '.txt'
    with open(path, 'a', encoding='utf-8') as f:
        f.write(str(data['argu']))
        f.write('\n')
    # question = '触发词为' + data['trigger'] + '的事件' + str(data['event_type']).split('-')[1] + '中角色' + data[
    #     'role'] + '是什么？'
    # sentence = data['text']
