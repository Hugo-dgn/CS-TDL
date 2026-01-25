import torch

def psi(y):
    idx = torch.abs(y) < 1
    z = torch.zeros_like(y)
    z[idx] = torch.exp(-1/(1-y[idx]**2))
    
    return z
    

class F:
    
    def __init__(self, eps, K1):
        self.eps = eps
        self.K1 = K1
        
    def __call__(self, y):
        return 1 / self.eps * psi((torch.norm(y, dim=1) - self.K1)/self.eps)


def mu(n, size):
    Z = torch.randn(size, n)
    U = Z / Z.norm(dim=1, keepdim=True)
    R = n*torch.rand(size, 1).pow(1/n)
    X = R * U
    return X
