import json

dev_path = './DuEE1.0/duee_dev.json'
test_path = './DuEE1.0/duee_test.json'
train_path = './DuEE1.0/duee_train.json'
label_path = './DuEE1.0/label.txt'
new_train_path = './DuEE1.0/new_duee_train.json'
schema_path = './DuEE1.0/duee_event_schema.json'

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
'''
with open(schema_path, 'r', encoding='utf-8') as s, open(label_path, 'w') as l:
    data = ['0 O']
    idx = 1
    for line in s.readlines():
        json_data = json.loads(line)
        data.append(str(idx) + ' ' + 'B-' + json_data['event_type'])
        idx += 1
        data.append(str(idx) + ' ' + 'I-' + json_data['event_type'])
        idx += 1
        for role in json_data['role_list']:
            data.append(str(idx) + ' ' + 'B-' + json_data['event_type'] + '-' + role['role'])
            idx += 1
            data.append(str(idx) + ' ' + 'I-' + json_data['event_type'] + '-' + role['role'])
            idx +=1
    for d in data:
        l.write(d + '\n')
'''

# 统计事件个数
'''
with open(train_path, 'r', encoding='utf-8') as f:
    type = {}
    for line in f.readlines():
        json_data = json.loads(line)
        for event in json_data['event_list']:
            if event['event_type'] in type:
                type[event['event_type']] += 1
            else:
                type[event['event_type']] = 1
    type = sorted(type.items(), key=lambda x: x[1])
    print(type)

with open('./event_type_num.txt','w') as f:
    for item in type:
        f.write(str(item[0])+'\t\t\t\t\t\t\t'+str(item[1])+'\n')
'''

# 修复数据
with open(train_path,'r',encoding='utf-8') as f,open(new_train_path,'w',encoding='utf-8') as nf:
    for line in f.readlines():
        json_data = json.loads(line)
        for event in json_data['event_list']:
            for argu in event['arguments']:
                if argu['argument'][0] == ' ':
                    argu['argument'] = str(argu['argument']).lstrip(' ')
                    argu['argument_start_index'] +=1
        nf.write(json.dumps(json_data,ensure_ascii= False) + '\n')



