import numpy as np
import torch
from transformers import AutoTokenizer

from data_preprocess import myDataSet
from var import device, id2label, dev_path, checkpoint

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

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = torch.load('./train_model/2.bin').to(device)
model.eval()
with torch.no_grad():
    dataset = myDataSet(dev_path)
    no_answer = 0
    for data in dataset:
        question = '触发词为' + data['trigger'] + '的事件' + str(data['event_type']).split('-')[1] + '中角色' + data[
            'role'] + '是什么？'
        sentence = data['text']

        inputs = tokenizer(question, sentence, truncation=True, return_tensors="pt", max_length=512,
                           return_offsets_mapping=True).to(device)

        mapping = inputs.pop('offset_mapping').squeeze(0)
        offset = (inputs['attention_mask'] - inputs['token_type_ids']).squeeze(0).sum().item()

        # 位置信息
        trigger_position = np.zeros(inputs['input_ids'].shape, dtype=int)
        trigger_start = inputs.char_to_token(data['trigger_start'], sequence_index=1) - 1
        trigger_end = inputs.char_to_token(data['trigger_end'], sequence_index=1) + 1
        second_seq_start = offset
        second_seq_end = inputs['attention_mask'].squeeze(0).sum().item() - 2
        count = 1


        mark = ['。', '？', '！', '；', '?', '!', ';']
        while trigger_start >= second_seq_start:
            trigger_position[0][trigger_start] = count
            if inputs.tokens()[trigger_start] in mark and count < 4:
                count += 1
            trigger_start -= 1
        count = 1

        while trigger_end <= second_seq_end:
            trigger_position[0][trigger_end] = count
            if inputs.tokens()[trigger_end] in mark and count < 4:
                count += 1
            trigger_end += 1
        _, pred = model(inputs, torch.tensor(trigger_position).to(device))
        pred = pred[0]

        idx = 0
        answer = []
        while idx < len(pred):
            if pred[idx] == 1:
                start, end = mapping[idx + offset]
                while idx + 1 < len(pred) and pred[idx + 1] == 2:
                    idx += 1
                _, end = mapping[idx + offset]
                answer.append(sentence[start:end])
            idx += 1
        if len(answer) == 0:
            no_answer += 1
        print(question)
        print(sentence)
        print(answer)
        print(data['argu'])
        print('\n')
    print(no_answer)

'''
触发词为塌陷的事件坍/垮塌中角色坍塌主体是什么？
7月4日，由中铁十九局承建的青岛地铁1号线胜利桥站施工围挡处发生塌陷，造成一名施工人员死亡；而在此之前的5月27日，由中铁二十局施工的地铁4号线沙子口静沙区间施工段坍塌，5名被困工人全部遇难。
['由中铁十九局承建的青岛地铁1号线胜利桥站施工围挡处', '中铁二十局施工的地铁4号线沙子口静沙区间施工段']
['青岛地铁1号线胜利桥站施工围挡处']
'''
