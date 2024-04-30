import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子以确保实验的可重复性
torch.manual_seed(42)
np.random.seed(42) # 哪里用到了？

# 定义Critic网络
class Critic(nn.Module): 
    def __init__(self, pid_dim):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(    # 返回一个函数指针，因为下面使用相应参数
            nn.Linear(pid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Critic网络输出一个价值
        )
        # 定义一个残差网络
        self.residual = nn.Linear(pid_dim, 1)
    
    def forward(self, x):   
        return self.layers(x) + self.residual(x)

# 初始化网络和优化器
input_dim =  1 # 无人机参数的维度，这里简化，只考虑无人机质量
pid_dim = 6 # PID参数的数量

# 将batched_inputs归一化到[0, 1]范围
kp_range = 15
ki_range = 15
kd_range = 5

# 创建一个包含所有PID参数范围的张量
pid_ranges = torch.tensor([kp_range, ki_range, kd_range, kp_range, ki_range, kd_range])

if __name__ == "__main__":

    # 读取CSV文件
    pid_parameters_df = pd.read_csv('/mnt/data/Training set/pid_parameters.csv')
    pid_values_df = pd.read_csv('/mnt/data/Training set/PID_Value.csv')

    # 确保CSV文件的列顺序一致，并且每一行对应一个样本
    pid_parameters = pid_parameters_df.values.astype(np.float32)
    pid_values = pid_values_df.values.astype(np.float32)

    # 转换为numpy数组，然后转换为Tensor
    pid_parameters_tensor = torch.tensor(pid_parameters)
    pid_values_tensor = torch.tensor(pid_values)

    # 初始化SummaryWriter
    writer = SummaryWriter('runs/critic_training')

    # 初始化一个Critic网络
    critic = Critic(pid_dim)   

    # 为这个Critic网络定义Adam优化器
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
    # critic.parameters()是从critic当中继承来的函数，每个Module都有，返回模型权重和偏置，以tensor的形式

    # 实例化学习率调度器
    scheduler = lr_scheduler.StepLR(critic_optimizer, step_size=50, gamma=0.5)

    # 定义和初始化DataLoader用于Critic网络的训练
    critic_dataloader = DataLoader(TensorDataset(pid_parameters_tensor, pid_values_tensor), batch_size=32, shuffle=True)
    # 这返回的是竟然是一个枚举类


    # 训练Critic网络
    num_epochs_critic = 200  # Critic网络的训练轮数 
    for epoch in range(num_epochs_critic):
        total_loss = 0  # 初始化总损失，用于整个epoch的累加
        for batch_idx, (batched_inputs, batched_values) in enumerate(critic_dataloader): # 每一个批次
            
            # 假设batched_inputs是一个形状为 (batch_size, 6) 的张量
            # 现在使用pid_ranges来归一化batched_inputs
            batched_inputs = batched_inputs / pid_ranges

            # Critic网络的训练过程...
            critic_values = critic(batched_inputs)
            critic_loss = nn.MSELoss()(critic_values, batched_values) # 定义损失函数，是利用梯度下降，训练网络的一个必要环节
            critic_optimizer.zero_grad() # 清除之前的梯度，能看出，梯度是保存在网络中的
            
            critic_loss.backward()  # 反向传播，计算当前的梯度，这个函数都没将需要计算损失函数的梯度的网络传进来
            critic_optimizer.step() # Adam优化器更新权重的函数

            # 累加损失
            total_loss += critic_loss.item()  

        # 计算epoch的平均损失
        average_loss = total_loss / len(critic_dataloader)

        # 打印当前epoch和平均损失
        print(f'Epoch {epoch + 1}/{num_epochs_critic}, Loss: {average_loss}')  
        
        # 使用TensorBoard记录Critic网络的损失...
        writer.add_scalar('Losses/Critic_Loss', critic_loss.item(), epoch * len(critic_dataloader) + batch_idx)

        # 展示模型参数的分布直方图
        for name, param in critic.named_parameters():
            writer.add_histogram(name, param, epoch * len(critic_dataloader) + batch_idx)

        # 更新学习率
        scheduler.step()

    # 关闭SummaryWriter
    writer.close()

    # 保存训练好的Critic网络
    torch.save(critic.state_dict(), 'critic.pth') # 这里保存的网络并不包括了param.requires_grad = False