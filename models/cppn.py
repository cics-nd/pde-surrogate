"""
Compositional pattern-producing network (CPPN)
(x, y) --> p
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CPPN(nn.Sequential):
    """size 1024 x 1024 is okay to generate at one pass
    """
    def __init__(self, dim_in, dim_out, dim_hidden, layers_hidden, act='tanh', 
        xavier_init=True):
        super(CPPN, self).__init__()

        self.add_module('fc0', nn.Linear(dim_in, dim_hidden, bias=None))
        self.add_module('act0', nn.Tanh())
        for i in range(1, layers_hidden):
            self.add_module('fc{}'.format(i), nn.Linear(dim_hidden, dim_hidden, bias=True))
            if act == 'tanh':
                self.add_module('act{}'.format(i), nn.Tanh())
            elif act == 'relu':
                self.add_module('act{}'.format(i), nn.ReLU())
            else:
                raise ValueError(f'unknown activation function: {act}')

        self.add_module('fc{}'.format(layers_hidden), nn.Linear(dim_hidden, dim_out))
        if xavier_init:
            self.init_xavier()
    
    def forward_test(self, x):
        print('{:<15}{:>30}'.format('input', str(x.size())))
        for name, module in self._modules.items():
            x = module(x)
            print('{:<15}{:>30}'.format(name, str(x.size())))
        return x

    def init_xavier(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            
    def _model_size(self):
        n_params, n_fc_layers = 0, 0
        for name, param in self.named_parameters():
            if 'fc' in name:
                n_fc_layers += 1
            n_params += param.numel()
        return n_params, n_fc_layers

        
def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    else:
        raise ValueError('Unknown activation function')

# densely fully connected NN

class _ResLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, act='tanh'):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=True)
        self.fc2 = nn.Linear(dim_hidden, dim_out, bias=True)
        if act == 'tanh':
            self.act = F.tanh 
        elif act == 'relu':
            self.act = F.relu

    def forward(self, x):
        # pre-activation
        res = x
        out = self.fc1(self.act(x))
        out = self.fc2(self.act(out))
        return res + out

class ResCPPN(nn.Sequential):

    def __init__(self, dim_in, dim_out, dim_hidden, res_layers, act='tanh'):
        super().__init__()
        self.add_module('fc0', nn.Linear(dim_in, dim_hidden, bias=None))

        for i in range(res_layers):
            reslayer = _ResLayer(dim_hidden, dim_hidden, dim_hidden, act=act)
            self.add_module(f'reslayer{i+1}', reslayer)
        
        self.add_module('act_last', activation(act))
        self.add_module('fc_last', nn.Linear(dim_hidden, dim_out, bias=True))

    def _model_size(self):
        n_params, n_fc_layers = 0, 0
        for name, param in self.named_parameters():
            if 'fc' in name:
                n_fc_layers += 1
            n_params += param.numel()
        return n_params, n_fc_layers
            

if __name__ == '__main__':

    cppn = ResCPPN(dim_in=2, dim_out=1, dim_hidden=64, res_layers=3, act='tanh')

    print(cppn)
    print(cppn._model_size())
    x = torch.randn(16, 2)
    y = cppn(x)

    print(y.shape)

        


