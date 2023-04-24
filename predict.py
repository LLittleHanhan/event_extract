import numpy as np
import torch
from transformers import AutoTokenizer

from var import device, id2label

sentence = '当地时间4月23日早晨，秘鲁前总统亚历杭德罗·托莱多自美国被引渡回到秘鲁,他将因腐败和洗钱指控接受秘鲁司法部门的审判 日前 秘鲁前总统亚历杭德罗·托莱多在其妻子和辩护人的陪同下向美国司法机构自首。托莱多于2001年至2006年任秘鲁总统。在其卸任后，秘鲁司法机构指控托莱多任内曾收受巴西大型建筑企业奥德布雷希特公司的巨额贿赂，帮助对方获取秘鲁公路建设工程合同。如果罪名成立，托莱多最高可被判处20年6个月监禁'

bert_tokenizer = AutoTokenizer.from_pretrained('./bert-base-chinese')
bert_model = torch.load('./train_model/6model.bin').to(device)
bert_model.eval()
with torch.no_grad():
    inputs = bert_tokenizer(sentence, truncation=True, return_tensors="pt", return_offsets_mapping=True)
    # print(inputs['offset_mapping'].squeeze(0).shape)
    offsets = inputs.pop('offset_mapping').squeeze(0)

    inputs = inputs.to(device)
    pred = bert_model(inputs)
    # print(pred.shape) # 1,x,131
    probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].cpu().numpy().tolist()
    # print(torch.nn.functional.softmax(pred,dim=-1)[0].shape) #x,131
    predictions = pred.argmax(dim=-1)[0].cpu().numpy().tolist()
    # print(pred.argmax(dim=-1)[0].shape) #40

    pred_label = []
    idx = 0
    while idx < len(predictions):
        pred = predictions[idx]
        label = id2label[pred]
        if label != "O":
            label = label[2:]
            start, end = offsets[idx]
            all_scores = [probabilities[idx][pred]]
            while idx + 1 < len(predictions) and id2label[predictions[idx + 1]] == f"I-{label}":
                all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = offsets[idx + 1]
                idx += 1
            score = np.mean(all_scores).item()
            start, end = start.item(), end.item()
            word = sentence[start:end]
            pred_label.append(
                {
                    "entity_group": label,
                    "score": score,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )
        idx += 1
    print(pred_label)
