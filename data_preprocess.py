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
            for line in f.readlines():
                json_data = json.loads(line)
                for event in json_data['event_list']:
                    # 去重
                    for argu in event['arguments']:
                        del argu['alias']
                    event['arguments'] = [dict(dic) for dic in
                                          set([tuple(argu.items()) for argu in event['arguments']])]

                    # 找相同的role
                    role_dic = {}

                    for argu in event['arguments']:
                        if argu['role'] in role_dic:

                            role_dic[argu['role']][0].append(argu['argument'])
                            role_dic[argu['role']][1].append(argu['argument_start_index'])
                            role_dic[argu['role']][2].append(argu['argument_start_index'] + len(argu['argument']) - 1)
                        else:
                            role_dic[argu['role']] = [[argu['argument']], [argu['argument_start_index']],
                                                      [argu['argument_start_index'] + len(argu['argument']) - 1]]

                    # 遍历role_dic
                    for k, v in role_dic.items():
                        dic = {'text': json_data['text'], 'event_type': event['event_type'],
                               'trigger': event['trigger'], 'trigger_start': event['trigger_start_index'],
                               'trigger_end': event['trigger_start_index'] + len(event['trigger']) - 1,
                               'role': k, 'argu': v[0],
                               'argu_start': v[1],
                               'argu_end': v[2]}
                        self.data.append(dic)
                    role_dic.clear()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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


def collote_fn(batch_samples):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    batch_question, batch_text = [], []
    for sample in batch_samples:
        question = '触发词为' + sample['trigger'] + '的事件' + str(sample['event_type']).split('-')[1] + '中角色' + sample[
            'role'] + '是什么？'
        batch_question.append(question)
        batch_text.append(sample['text'])
    batch_inputs = tokenizer(
        batch_question,
        batch_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    trigger_position = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for idx, (question, text) in enumerate(zip(batch_question, batch_text)):
        encoding = tokenizer(question, text, truncation=True, max_length=512)
        for (start, end) in zip(batch_samples[idx]['argu_start'], batch_samples[idx]['argu_end']):
            token_start = encoding.char_to_token(start, sequence_index=1)
            token_end = encoding.char_to_token(end, sequence_index=1)
            batch_label[idx][token_start] = label2id['B']
            batch_label[idx][token_start + 1:token_end + 1] = label2id['I']

        # 加入位置信息
        trigger_start = encoding.char_to_token(batch_samples[idx]['trigger_start'], sequence_index=1) - 1
        trigger_end = encoding.char_to_token(batch_samples[idx]['trigger_end'], sequence_index=1) + 1
        second_seq_start = encoding.char_to_token(0, sequence_index=1)
        second_seq_end = encoding.char_to_token(len(batch_samples[idx]['text']) - 1, sequence_index=1)
        count = 1
        while trigger_start >= second_seq_start:
            trigger_position[idx][trigger_start] = count
            count += 1
            trigger_start -= 1
        count = 1
        while trigger_end <= second_seq_end:
            trigger_position[idx][trigger_end] = count
            count += 1
            trigger_end += 1

    return batch_inputs, torch.tensor(batch_label), torch.tensor(trigger_position)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from var import dev_path, train_path

    dev_data = myDataSet(dev_path)
    dev_dataloader = DataLoader(dev_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    train_data = myDataSet(train_path)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    print(len(dev_dataloader))
    print(len(train_dataloader))

    it = iter(train_dataloader)
    while True:
        batch_X, batch_y, batch_z = next(it)
        from torch.utils.tensorboard import SummaryWriter
        from model import myBert

        model = myBert.from_pretrained('./bert-base-chinese')
        writer = SummaryWriter('./tensorboard')
        writer.add_graph(model,
                         input_to_model=[
                             {'input_ids': batch_X['input_ids'], 'attention_mask': batch_X['attention_mask'],
                              'token_type_ids': batch_X['token_type_ids']}, batch_z, batch_y])
        # print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
        # print('batch_y shape:', batch_y.shape)
        # print(batch_X)
        print(batch_y)
        print(batch_z)
        break
