import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# create base conv block with conv2d, bn, and relu activation.

def get_activation(atype):

    if atype=='relu':
        nonlinear = nn.ReLU()
    elif atype=='tanh':
        nonlinear = nn.Tanh() 
    elif atype=='sigmoid':
        nonlinear = nn.Sigmoid() 
    elif atype=='elu':
        nonlinear = nn.ELU()

    return nonlinear

def makeblock_conv(in_chs, out_chs, atype, stride=1):

    layer = nn.Conv2d(in_channels=in_chs, 
        out_channels=out_chs, kernel_size=5, stride=stride)
    bn = nn.BatchNorm2d(out_chs, affine=False) # Batch Norm not learnable #Hence no params when we extract hidden params of the model
    nonlinear = get_activation(atype)

    return nn.Sequential(*[layer, bn, nonlinear])

def makeblock_dense(in_dim, out_dim, atype):
    
    layer = nn.Linear(in_dim, out_dim)
    bn = nn.BatchNorm1d(out_dim, affine=False) # Batch Norm not learnable #Hence no params when we extract hidden params of the model
    nonlinear = get_activation(atype)
    out = nn.Sequential(*[layer, bn, nonlinear])
    
    return out

class ModelConv(nn.Module):

    def __init__(self, in_width=784, hidden_width=64, n_layers=5, atype='relu', 
        last_hidden_width=None, data_code='cifar10', **kwargs):
        super(ModelConv, self).__init__()
    
        block_list = []
        is_conv = False

        if data_code == 'cifar10':
            in_ch = 3
        elif data_code == 'mnist':
            in_ch = 1

        last_hw = hidden_width
        if last_hidden_width:
            last_hw = last_hidden_width
        
        for i in range(n_layers):
            block = makeblock_conv(hidden_width, hidden_width, atype)
            block_list.append(block)

        self.input_layer    = makeblock_conv(in_ch, hidden_width, atype)
        self.sequence_layer = nn.Sequential(*block_list)
        if data_code == 'mnist':
            dim = 2048
        elif data_code == 'cifar10':
            dim = 8192

        self.output_layer = makeblock_dense(dim, last_hw, atype)

        self.is_conv = is_conv
        self.in_width = in_width

    def forward(self, x):

        output_list = []
        
        x = self.input_layer(x)
        output_list.append(x)
        
        for block in self.sequence_layer:
            x = block(x.clone())
            output_list.append(x)
            
        x = x.view(-1, np.prod(x.size()[1:]))

        x = self.output_layer(x)
        output_list.append(x)

        return x, output_list


class ModelVanilla(nn.Module):

    def __init__(self, hidden_width=64, last_hidden_width=None, **kwargs):
        super(ModelVanilla, self).__init__()
        last_dim = hidden_width
        if last_hidden_width:
            last_dim = last_hidden_width
            
        self.output = nn.Linear(last_dim, 10)

    def forward(self, x):
        x = self.output(x)
        return F.log_softmax(x, dim=1)


class ModelEnsemble(nn.Module):

    def __init__(self, hsic_model, vanilla_model):
        super(ModelEnsemble, self).__init__()

        self._hsic_model = hsic_model
        self._vanilla_model = vanilla_model
        
    def forward(self, x):
        
        x, hiddens = self._hsic_model(x)
        x = self._vanilla_model(x)
        return x, hiddens