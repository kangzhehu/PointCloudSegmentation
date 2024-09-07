
from utils.scheduler import WarmupCosineLR

import torch
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR


# class WarmupCosineLR(LambdaLR):
#     def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0, last_epoch=-1):
#         self.T_max = T_max
#         self.eta_min = eta_min
#         self.warmup_steps = warmup_steps
#         self.initial_lr = optimizer.param_groups[0]['lr']
#
#         if T_max <= warmup_steps:
#             raise ValueError("T_max must be greater than warmup_steps")
#
#         def lr_lambda(step):
#             if step < self.warmup_steps:
#                 print(self.initial_lr)
#                 # Warmup phase: increase from 0 to 1 (proportion of initial_lr)
#                 return step / self.warmup_steps
#             else:
#                 # Cosine annealing phase
#                 print(self.initial_lr)
#                 cosine_step = step - self.warmup_steps
#                 cosine_total_steps = self.T_max - self.warmup_steps
#                 return self.eta_min / self.initial_lr + 0.5 * (1 - self.eta_min / self.initial_lr) * (
#                             1 + math.cos(math.pi * cosine_step / cosine_total_steps))
#
#         super(WarmupCosineLR, self).__init__(optimizer, lr_lambda, last_epoch)


# 定义一个简单的模型
model = torch.nn.Linear(10, 1)

# 初始学习率
initial_lr = 0.001

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# 参数设置
total_steps = 40 * 100  # 总步数
warmup_steps = 3 * 100  # warmup 步数
eta_min = 0.00001  # 最小学习率

# 调度器
scheduler = WarmupCosineLR(optimizer, T_max=total_steps, eta_min=eta_min, warmup_steps=warmup_steps)

# 存储学习率变化
lrs = []

# 模拟训练循环
for step in range(total_steps):
    # 模拟一个训练步骤
    optimizer.step()
    # 存储当前学习率
    lrs.append(optimizer.param_groups[0]['lr'])
    # print(optimizer.param_groups[0]['lr'])
    # 更新调度器
    scheduler.step()


# 可视化学习率变化
plt.plot(lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()