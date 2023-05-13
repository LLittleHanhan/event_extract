import torch

# 文件路径
dev_path = './class/new_dev.txt'
test_path = './class/new_test.txt'
train_path = './class/new_train.txt'
label_path = './class/label.txt'
test_result_path = './class/result.txt'
# 超参
checkpoint = './bert-base-chinese'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


