import numpy as np
import torch
from transformers import AutoTokenizer

from var import device, bert_model
from data_preprocess import id2label

sentence = '廖浩'

bert_tokenizer = AutoTokenizer.from_pretrained('./bert-base-chinese')
bert_model.load_state_dict(
    torch.load('./train_model/model.bin', map_location=torch.device(device))
)
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