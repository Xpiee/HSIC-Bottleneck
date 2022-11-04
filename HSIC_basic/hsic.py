import torch
import numpy as np

# This is the main implementation of Equation 3 of the paper

## Implement Gausian kernel function to calculate K_X and K_y
## gausian kernel, k(x, y) ~ exp(-(1/2)*||x - y||^2/sigma**2 )

def distmat(X):
    """ Math for calculating distance matrix
        Implementing Euclidean Distance Matrix (EDM)
        D = abs(a^2 + b^2 - 2a.b_T)
        Used in generating kernel matrix in kernelmat()
    """
    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D)

    return D
    
# https://github.com/choasma/HSIC-bottleneck/blob/9f1fe2447592d61c0ba524aad0ff0820ae2ba9cb/source/hsicbt/core/
def kernelmat(X, sigma):
    """
    Kernel function

    m: training batch size
    H: centering matrix:: I_m - (1/m)*1_m.1_m
    gaussian kernel: k(x, y) ~ exp(-(1/2)*||x - y||^2/sigma**2)
    """
    m = int(X.size()[0]) # batch size
    H = torch.eye(m) - (1./m) * torch.ones([m,m])

    Dxx = distmat(X)
    variance = 2.*sigma*sigma*X.size()[1]            
    Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)   # Gaussian kernel
    Kxc = torch.mm(Kx, H) # kernel function centered with H

    return Kxc

def hsic_base(x, y, sigma=None, use_cuda=True):
    """
    Implement equation 3 in the paper
    HSIC: (m - 1)^-2 . trace(Kx H Ky H)
    """
    m = int(x.size()[0]) # batch size

    KxH = kernelmat(x, sigma=sigma)
    KyH = kernelmat(y, sigma=sigma)

    return (torch.trace(KxH @ KyH))/(m - 1)**2

# From HSIC implementation 
# https://github.com/choasma/HSIC-bottleneck/blob/9f1fe2447592d61c0ba524aad0ff0820ae2ba9cb/source/hsicbt/core/train_misc.py#L26

def hsic_loss_obj(hidden, h_target, h_data, sigma):
    """
    calculate hsic between input (X) and hidden layer weights
    calculate hsic between hidden layer weights and target (Y)

    return: hx, hy for calculating loss in training pipeline
    """
    hsic_hx = hsic_base(hidden, h_data, sigma=sigma)
    hsic_hy = hsic_base(hidden, h_target, sigma=sigma)

    return hsic_hx, hsic_hy