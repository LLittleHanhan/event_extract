import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_scheduler

# initial_lr = 0.1
#
# class model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
#
#     def forward(self, x):
#         pass
#
# net_1 = model()
#
# optimizer_1 = torch.optim.Adam(net_1.parameters(), lr = initial_lr)
# scheduler_1 = LambdaLR(optimizer_1, lr_lambda=lambda epoch: 1/(epoch+1))
#
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer_1,
#     num_warmup_steps=0,
#     num_training_steps=2,
# )
#
# print("初始化的学习率：", optimizer_1.defaults['lr'])
#
# for epoch in range(1, 11):
#     # train
#     optimizer_1.zero_grad()
#     optimizer_1.step()
#     print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
#     lr_scheduler.step()
#     # scheduler_1.step()


# x = torch.randn(2, 2)
# x.requires_grad = True
#
# lin0 = nn.Linear(2, 2, bias=False)
# lin1 = nn.Linear(2, 2, bias=False)
# lin2 = nn.Linear(2, 2, bias=False)

# print('什么都没有')
# x1 = lin0(x)
# # with torch.no_grad():
# x2 = lin1(x1)
# x3 = lin2(x2)
# x3.sum().backward()
# print(lin0.weight.grad)
# print(lin1.weight.grad)
# print(lin2.weight.grad)

# print('with torch.no_grad()')
# x1 = lin0(x)
# with torch.no_grad():
#     x2 = lin1(x1)
# x3 = lin2(x2)
# x3.sum().backward()
# print(lin0.weight.grad)
# print(lin1.weight.grad)
# print(lin2.weight.grad)
# print('x,x1,x2,x3')
# print('x',x.grad)
# print('x1',x1.grad)
# print('x2',x2.grad)
# print('x3',x3.grad)
# print(x.is_leaf)
# print(x1.is_leaf)
# print(x2.is_leaf)
# print(x3.is_leaf)
# print(lin2.weight.is_leaf)

# print('requires grad')
# lin2.weight.requires_grad = False
# x1 = lin0(x)
# x2 = lin1(x1)
# x3 = lin2(x2)
# x3.sum().backward()
# print(lin0.weight.grad)
# print(lin1.weight.grad)
# print(lin2.weight.grad)
# print('\n')
# print('x,x1,x2,x3')
# print('x',x.grad)
# print('x1',x1.grad)
# print('x2',x2.grad)
# print('x3',x3.grad)
# print('\n')
# print(x.is_leaf)
# print(x1.is_leaf)
# print(x2.is_leaf)
# print(x3.is_leaf)
# print('\n')
# print(lin0.weight.is_leaf)
# print(lin1.weight.is_leaf)
# print(lin2.weight.is_leaf)


# x = torch.tensor([2., 1.], requires_grad=True)
# y = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
#
# z = torch.mm(x.view(1, 2), y)
# print(f"z:{z}")
# z.backward(torch.Tensor([[1., 0]]), retain_graph=True)
# print(f"x.grad: {x.grad}")
# print(f"y.grad: {y.grad}")


import matplotlib.pyplot as plt

#折线图
x = [5,7,11,17,19,25]#点的横坐标
k1 = [0.8222,0.918,0.9344,0.9262,0.9371,0.9353]#线1的纵坐标
k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.plot(x,k1,'s-',color = 'r',label="ATT-RLSTM")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("region length")#横坐标名字
plt.ylabel("accuracy")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
