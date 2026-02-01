import torch
import math

def g(y, m):
    idx = torch.logical_and(y >= 0, y <= 1)
    z = torch.zeros_like(y)
    z[idx] = 4**(m+1) * y[idx]**(m+1) * (1-y[idx])**(m+1)
    Cm = 4**(m+1) * math.factorial(m+1)**2 / math.factorial(2*m+3)
    return z / Cm
    

class F:
    def __init__(self, eps, K1, m):
        self.eps = eps
        self.K1 = K1
        self.m = m
        
    def __call__(self, x):
        y = torch.norm(x, dim=1)
        z = 1 / self.eps * g((y - self.K1)/self.eps, self.m)
        return z


def mu(n, size):
    Z = torch.randn(size, n)
    U = Z / Z.norm(dim=1, keepdim=True)
    R = n*torch.rand(size, 1).pow(1/n)
    X = R * U
    return X
