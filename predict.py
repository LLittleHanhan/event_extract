import numpy as np
import torch
from transformers import AutoTokenizer

from var import device, checkpoint

class_list = ['finance', 'stocks', 'education', 'science', 'society', 'politics', 'sports', 'game', 'entertainment']
title = '雅思写作素材：科技话题必备短语'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = torch.load('./train_model/class.bin').to(device)
model.eval()
with torch.no_grad():
    inputs = tokenizer(title, truncation=True, return_tensors="pt").to(device)
    id = model(inputs)[0].argmax(dim=-1).cpu().numpy().tolist()[0]
    print(class_list[id])
