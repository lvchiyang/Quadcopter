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
np.random.seed(42)

# 定义Critic网络
class Critic(nn.Module): 
    def __init__(self, pid_dim):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(pid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Critic网络输出一个价值
        )
        # 定义一个残差网络
        self.residual = nn.Linear(pid_dim, 1)
    
    def forward(self, x):   
        return self.layers(x) + self.residual(x)

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化网络
critic = Critic(pid_dim=6).to(device)  # 确保模型在GPU上

# 读取CSV文件
pid_parameters_df = pd.read_csv('/mnt/data/Training set/pid_parameters.csv')
pid_values_df = pd.read_csv('/mnt/data/Training set/PID_Value.csv')

# 将数据转换为Tensor，并迁移到GPU
pid_parameters = pid_parameters_df.values.astype(np.float32)
pid_values = pid_values_df.values.astype(np.float32)
pid_parameters_tensor = torch.tensor(pid_parameters, dtype=torch.float32).to(device)
pid_values_tensor = torch.tensor(pid_values, dtype=torch.float32).to(device)

# 将batched_inputs归一化到[0, 1]范围
kp_range = 15
ki_range = 15
kd_range = 5
pid_ranges = torch.tensor([kp_range, ki_range, kd_range, kp_range, ki_range, kd_range], dtype=torch.float32).to(device)

# 初始化SummaryWriter
writer = SummaryWriter('runs/critic_training')

# 初始化DataLoader
critic_dataloader = DataLoader(TensorDataset(pid_parameters_tensor, pid_values_tensor), batch_size=32, shuffle=True)

# 为Critic网络定义Adam优化器
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# 实例化学习率调度器
scheduler = lr_scheduler.StepLR(critic_optimizer, step_size=100, gamma=0.5)

# 训练Critic网络
num_epochs_critic = 500
for epoch in range(num_epochs_critic):
    total_loss = 0.0
    critic.train()  # 设置模型为训练模式
    for batch_idx, (batched_inputs, batched_values) in enumerate(critic_dataloader):
        # 归一化batched_inputs
        batched_inputs = batched_inputs / pid_ranges
        
        # 前向传播
        critic_values = critic(batched_inputs)
        
        # 计算损失
        critic_loss = nn.MSELoss()(critic_values, batched_values)
        
        # 反向传播和优化
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # 累加损失
        total_loss += critic_loss.item()

    # 计算epoch的平均损失
    average_loss = total_loss / len(critic_dataloader)
    
    # 打印当前epoch和平均损失
    print(f'Epoch {epoch + 1}/{num_epochs_critic}, Loss: {average_loss}')
    
    # 使用TensorBoard记录Critic网络的损失
    writer.add_scalar('Losses/Critic_Loss', average_loss, global_step=epoch)

    # 更新学习率
    scheduler.step()

    # # 展示模型参数的分布直方图
    # for name, param in critic.named_parameters():
    #     writer.add_histogram(name, param.data, global_step=epoch)

# 关闭SummaryWriter
writer.close()

# 保存训练好的Critic网络
torch.save(critic.state_dict(), 'critic.pth')