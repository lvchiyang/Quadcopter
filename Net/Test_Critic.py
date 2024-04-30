import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter

import Critic_Net


# 假设这里是测试集数据的路径
test_pid_parameters_df = pd.read_csv('/mnt/data/Test set/Test_pid_parameters.csv')
test_pid_values_df = pd.read_csv('/mnt/data/Test set/Test_PID_Value.csv')

# 将测试集数据转换为Tensor
test_pid_parameters = test_pid_parameters_df.values.astype(np.float32)
test_pid_values = test_pid_values_df.values.astype(np.float32)
test_pid_parameters_tensor = torch.tensor(test_pid_parameters)
test_pid_values_tensor = torch.tensor(test_pid_values)

# 初始化SummaryWriter
writer = SummaryWriter('runs/critic_testing')

# 测试集的DataLoader
test_dataloader = DataLoader(TensorDataset(test_pid_parameters_tensor, test_pid_values_tensor), batch_size=10)

# 加载训练好的模型（确保模型已经训练完成并且有可用的模型权重）
critic = Critic_Net.Critic(Critic_Net.pid_dim)
critic.load_state_dict(torch.load('/mnt/critic.pth'))
critic.eval()  # 设置模型为评估模式

# 测试集上的推理
test_loss = 0
with torch.no_grad():
    for batched_inputs, batched_values in test_dataloader:
        batched_inputs = batched_inputs / Critic_Net.pid_ranges
        predictions = critic(batched_inputs) # 模型对于当前批次输入的预测
        test_loss += nn.MSELoss()(predictions, batched_values).item() # .item() 是将损失张量（通常是一个包含单个元素的零维张量）转换为Python的标量值。

    test_loss /= len(test_dataloader)
    print(f'Test Loss: {test_loss}')
    writer.add_scalar('Losses/Test_Loss', test_loss, global_step=0)

# # 记录额外的测试指标，例如R^2分数
# test_mse = mean_squared_error(test_pid_values, predictions.numpy())
# print(f'Test MSE: {test_mse}')
# writer.add_scalar('Metrics/Test_MSE', test_mse, global_step=0)

# 关闭SummaryWriter
writer.close()