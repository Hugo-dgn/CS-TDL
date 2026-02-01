import torch.nn as nn
import math

class OneHiddenNN(nn.Module):
    def __init__(self, n, params_count, output_size=1):
        super().__init__()
        hidden_size = max(1, (params_count - output_size) // (n + output_size))
        self.net = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

class TwoHiddenNN(nn.Module):
    def __init__(self, n, params_count, output_size=1):
        super().__init__()
        h = int((- (n + 2 + output_size) + math.sqrt((n + 2 + output_size)**2 + 4*(params_count - output_size))) // 2)
        h = max(1, h)
        self.net = nn.Sequential(
            nn.Linear(n, h),
            nn.Sigmoid(),
            nn.Linear(h, h),
            nn.Sigmoid(),
            nn.Linear(h, output_size)
        )
    
    def forward(self, x):
        return self.net(x)