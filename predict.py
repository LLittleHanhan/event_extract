import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig

from model import myBert
from var import device, id2label

sentence = '它死了'

tokenizer = AutoTokenizer.from_pretrained('./chinese-roberta-wwm-ext')
config = AutoConfig.from_pretrained('./chinese-roberta-wwm-ext')
model = myBert(config)
model.load_state_dict(torch.load('./train_model/1triggerModel.bin'))
model.eval()
with torch.no_grad():
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt", return_offsets_mapping=True, max_length=512)
    offsets = inputs.pop('offset_mapping').squeeze(0)
    _, pred = model(inputs)
    print(pred)
    for idx,token in enumerate(pred):


