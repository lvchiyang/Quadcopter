import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from Critic_Net import Critic

# 设置随机种子以确保实验的可重复性
torch.manual_seed(42)

# 将batched_inputs归一化到[0, 1]范围
kp_range = 15
ki_range = 15
kd_range = 5

# 创建一个包含所有PID参数范围的张量
pid_ranges = torch.tensor([kp_range, ki_range, kd_range, kp_range, ki_range, kd_range])

# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, input_dim, pid_dim, critic):
        super(ActorCritic, self).__init__()
        self.actor_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, pid_dim)  # Actor网络输出PID参数
                            # 这里应该为生成的，限制一个范围，或者归一化，
        ) 
    
    def forward(self, x):
        # Actor网络输出PID参数
        pid_params = self.actor_layers(x)
        # 使用复制的Critic网络评估动作的价值
        critic_value = critic(pid_params / pid_ranges)
        return pid_params, critic_value

# 初始化网络和优化器
input_dim =  1 # 无人机参数的维度，这里简化，只考虑无人机质量
pid_dim = 6 # PID参数的数量

# 初始化SummaryWriter
writer = SummaryWriter('runs/actor_critic_training')

# 在这里读取训练好的critic网络
critic = ActorCritic(input_dim, pid_dim, Critic)  
critic.load_state_dict(torch.load('critic.pth'))
critic.eval()

# # 现在Critic网络已经训练好，我们将固定其参数
# for param in critic.parameters(): # 可迭代对象，返回每一个张量
#     param.requires_grad = False    # 是针对每一行张量，告诉Pytorch在构建计算图的时候，不记录其梯度

# 定义Actor-Critic网络实例
actor_critic = ActorCritic(input_dim, pid_dim, critic)

# 定义和初始化DataLoader用于Actor网络的训练，这里不对，应该是无人机参数，这里简化，只考虑无人机质量m
# actor_dataloader = DataLoader(TensorDataset(pid_parameters_tensor), batch_size=64, shuffle=True)

# 只训练Actor部分
actor_optimizer = optim.Adam(actor_critic.parameters(), lr=0.001)

# 实例化学习率调度器
scheduler = lr_scheduler.StepLR(actor_optimizer, step_size=20, gamma=0.1)

m = 1
m_tensor = torch.tensor([m], dtype=torch.float32)

num_epochs_actor = 60  # Actor网络的训练轮数
for epoch in range(num_epochs_actor):
    # 从Actor网络获取PID参数
    actor_pid_params, critic_values = actor_critic(m_tensor) # 还能以这种方式接收函数的返回值
    
    # 计算Actor网络的损失
    actor_value = -critic_values.mean()  # 使用负的价值，因为我们希望最大化价值
    # 梯度下降，越来越负，吗？
    
    # 更新Actor网络的参数
    actor_optimizer.zero_grad()
    actor_value.backward() # 利用网络模型去计算梯度
    actor_optimizer.step()

    print(f'actor value: {actor_value}')
    
    # 使用TensorBoard记录Actor网络的损失...
    writer.add_scalar('Value/actor_value', actor_value.item(), epoch)

    # 更新学习率
    scheduler.step()

    # 展示模型参数的分布直方图
    for name, param in actor_critic.named_parameters():
        writer.add_histogram(name, param, epoch)

# 关闭SummaryWriter
writer.close()

# 保存训练好的Actor-Critic网络
torch.save(actor_critic.state_dict(), 'actor_critic.pth')