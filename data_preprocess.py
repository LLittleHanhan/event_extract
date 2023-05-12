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

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from var import label2id, checkpoint


class myDataSet(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                json_data = json.loads(l)
                dic = {"text": json_data["text"]}
                tags = []
                for event in json_data["event_list"]:
                    event_tag = {"event_type": event["event_type"], "trigger": event["trigger"],
                                 "start": event["trigger_start_index"],
                                 "end": len(event["trigger"]) + event["trigger_start_index"] - 1}
                    tags.append(event_tag)
                dic["tags"] = tags
                self.data.append(dic)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


'''
{'text': '消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了', 
 'tags': [
            {''event_type'': '组织关系-裁员', 'trigger': '裁员', 'start': 15, 'end': 17}
            {                                                                     }
         ]
}
'''


def collote_fn(batch_samples):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    batch_text, batch_tags = [], []
    for sample in batch_samples:
        batch_text.append(sample['text'])
        batch_tags.append(sample['tags'])
    batch_inputs = tokenizer(
        batch_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for t_idx, text in enumerate(batch_text):
        encoding = tokenizer(text, truncation=True, max_length=512)
        batch_label[t_idx][0] = -100
        batch_label[t_idx][len(encoding.tokens()) - 1:] = -100
        for tag in batch_tags[t_idx]:
            trigger_token_start = encoding.char_to_token(tag['start'])
            trigger_token_end = encoding.char_to_token(tag['end'])

            batch_label[t_idx][trigger_token_start] = label2id[f"B-{tag['event_type']}"]
            batch_label[t_idx][trigger_token_start + 1:trigger_token_end + 1] = label2id[f"I-{tag['event_type']}"]

    return batch_inputs, torch.tensor(batch_label, dtype=torch.long)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from var import dev_path, train_path

    dev_data = myDataSet(dev_path)
    dev_dataloader = DataLoader(dev_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    train_data = myDataSet(train_path)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)

    it = (iter(train_dataloader))
    while True:
        # batch_X, batch_y = next(it)
        # print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
        # print('batch_y shape:', batch_y.shape)
        # print(batch_X)
        # print(batch_y)
        print(next(it)[1])
        break
