import json

dev_path = './DuEE1.0/duee_dev.json'
test_path = './DuEE1.0/duee_test.json'
train_path = './DuEE1.0/duee_train.json'
label_path = './DuEE1.0/label.txt'
new_train_path = './DuEE1.0/new_duee_train.json'
schema_path = './DuEE1.0/duee_event_schema.json'

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
import json

with open(train_path, 'r', encoding='utf-8') as f, open(new_train_path, 'w',encoding="utf8") as nf:
    for l in f.readlines():
        json_data = json.loads(l)
        for event in json_data['event_list']:
            event['trigger'].strip()
        json_str = json.dumps(json_data,ensure_ascii=False)
        nf.write(json_str+'\n')
'''
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

with open(schema_path, 'r', encoding='utf-8') as s, open(label_path, 'w') as l:
    data = ['0 O']
    idx = 1
    for line in s.readlines():
        json_data = json.loads(line)
        data.append(str(idx) + ' ' + 'B-' + json_data['event_type'])
        idx += 1
        data.append(str(idx) + ' ' + 'I-' + json_data['event_type'])
        idx += 1
    for d in data:
        l.write(d+'\n')
