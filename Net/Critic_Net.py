import torch
import torch.nn as nn
# 定义Critic网络
class Critic(nn.Module): 
    def __init__(self, pid_dim):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(    # 返回一个函数指针，因为下面使用相应参数
            nn.Linear(pid_dim, 64), # 好家伙网络接受的是张量，因为后面一个batch用到
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