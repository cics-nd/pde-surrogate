r"""Mutliscale conditional Glow.

Instead of data-driven training, the forward path is from x --> z (encoding),
in equation-driven training the forward path is z --> x (generation/sampling).

Initialization:
    - ActNorm: identity transform
    - Invertible 1x1 Conv: random rotation matrix
    - Affine Coupling: Identity transform

Tricks to imporve the statbility of sampling-based training:
    - Clamp the standard deviation of the Guassian for all latent variables.

"""

import math
from collections import OrderedDict
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.clip_grad import clip_grad_value_
from models.codec import _DenseBlock, _DenseLayer, _Transition, module_size


class _DenseBlockInput(nn.Sequential):
    """For input dense block, feature map size the same as input"""
    def __init__(self, num_layers, in_features, init_features, growth_rate, 
                drop_rate, bn_size=4, bottleneck=False):
        super(_DenseBlockInput, self).__init__()
        self.num_layers = num_layers
        self.add_module('in_conv', nn.Conv2d(in_features, init_features-1, 
            kernel_size=3, stride=1, padding=1))
        
        for i in range(num_layers-1):
            layer = _DenseLayer(init_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module(f'denselayer{i+1}', layer)

    def forward(self, x):
        out = self.in_conv(x)
        out = torch.cat((x, out), 1)
        for i in range(self.num_layers-1):
            out = self[i+1](out)
        return out
        

class ActNorm(nn.Module):
    """Activation normalization, two ways to initialize:
        - data init: one minibatch of data
        - identity transform: used in sampling-based training of cGlow

    Args:
        in_features (Tensor): Number of input features
        return_logdet (bool): default True.
        data_init (bool): Use one minibatch data initialization or not, 
            default False.
    """
    def __init__(self, in_features, return_logdet=True, data_init=False):
        super(ActNorm, self).__init__()
        # identify transform
        self.weight = Parameter(torch.ones(in_features, 1, 1))
        self.bias = Parameter(torch.zeros(in_features, 1, 1))
        self.data_init = data_init
        self.data_initialized = False
        self.return_logdet = return_logdet

    def _init_parameters(self, input):
        # input: initial minibatch data
        # mean per channel: (B, C, H, W) --> (C, B, H, W) --> (C, BHW)
        input = input.transpose(0, 1).contiguous().view(input.shape[1], -1)
        mean = input.mean(1)
        std = input.std(1) + 1e-6
        self.bias.data = -(mean / std).unsqueeze(-1).unsqueeze(-1)
        self.weight.data = 1. / std.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        if self.data_init and (not self.data_initialized):
            self._init_parameters(x)
            self.data_initialized = True
        if self.return_logdet:
            logdet = self.weight.abs().log().sum() * x.shape[-1] * x.shape[-2]
            return self.weight * x + self.bias, logdet
        else:
            return self.weight * x + self.bias

    def reverse(self, y):
        if self.return_logdet:
            logdet = self.weight.abs().log().sum() * y.shape[-1] * y.shape[-2]
            return (y - self.bias) / self.weight, logdet
        else:
            return (y - self.bias) / self.weight


class InvertibleConv1x1(nn.Module):
    """Invertible 1x1 Conv layer.
    
    Forward path is still from x to z. But the matrix has to inverse.
    For sampling, the matrix does not need to inverse. 
    For this model, z --> x is used to train the model.
    Register one weight matrix and only update this one for both forward and 
    inverse.
    
    TODO mixed precision supported in pytorch? matrix inversion in float64,
    convert to float32 for the rest.

    Args:
        in_channels (int):
        train_sampling (bool): sampling is used for training Reverse KL loss,
            otherwise encoding is used for training forward KL loss
    """
    def __init__(self, in_channels, train_sampling=True):
        super(InvertibleConv1x1, self).__init__()
        
        dtype = np.float32
        w_shape = (in_channels, in_channels)
        # only one copy for both forward and reverse, 
        # depends on `training_sampling`, the inverse happens on less used path
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(dtype)

        self.w_shape = w_shape
        self.train_sampling = train_sampling
        self.weight = nn.Parameter(torch.Tensor(w_init))
    
    def forward(self, x):
        # x --> z
        # torch.slogdet() is not stable
        if self.train_sampling:
            W = torch.inverse(self.weight.double()).float()  
        else:
            W = self.weight
        logdet = self.log_determinant(x, W)        
        kernel = W.view(*self.w_shape, 1, 1)
        return F.conv2d(x, kernel), logdet

    def reverse(self, z):
        # z --> x
        if self.train_sampling:
            W = self.weight
        else:
            W = torch.inverse(self.weight.double()).float()  
        logdet = self.log_determinant(z, W)
        kernel = W.view(*self.w_shape, 1, 1)
        # negative logdet, since we are still computing p(x|z)
        return F.conv2d(z, kernel), -logdet

    def log_determinant(self, x, W):
        h, w = x.shape[2:]
        det = torch.det(W.to(torch.float64)).to(torch.float32)
        if det.item() == 0:
            det += 1e-6
        return h * w * det.abs().log()



class InvertibleConv1x1LU(nn.Module):
    """Invertible 1x1 Conv layer with LU decomposition.
    
    Forward path is still from x to z. But the matrix has to inverse.
    For sampling, the matrix does not need to inverse. 
    For this model, z --> x is used to train the model.
    Register one weight matrix and only update this one for both forward and 
    inverse.
    
    TODO mixed precision supported in pytorch?
    Args:
        in_channels (int):
        train_sampling (bool): sampling is used for training Reverse KL loss,
            otherwise encoding is used for training forward KL loss
    """
    def __init__(self, in_channels, train_sampling=True):
        super().__init__()    
        
        dtype = np.float32
        w_shape = (in_channels, in_channels)
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(dtype)

        self.w_shape = w_shape
        self.train_sampling = train_sampling
    
        p_np, l_np, u_np = scipy.linalg.lu(w_init)
        s_np = np.diag(u_np)
        sign_s_np = np.sign(s_np)
        log_s_np = np.log(abs(s_np))
        u_np = np.triu(u_np, k=1)
        l_mask = np.tril(np.ones_like(w_init), -1)
        u_mask = np.triu(np.ones_like(w_init), k=1)
        eye = np.eye(*w_shape, dtype=dtype)

        self.register_buffer('p', torch.Tensor(p_np.astype(dtype)))
        self.l = nn.Parameter(torch.Tensor(l_np.astype(dtype)))
        self.u = nn.Parameter(torch.Tensor(u_np.astype(dtype)))
        self.log_s = nn.Parameter(torch.Tensor(log_s_np.astype(dtype)))
        self.register_buffer('sign_s', torch.Tensor(sign_s_np.astype(dtype)))
        self.register_buffer('l_mask', torch.Tensor(l_mask))
        self.register_buffer('u_mask', torch.Tensor(u_mask))
        self.register_buffer('eye', torch.Tensor(eye))


    def weight(self):
        l = self.l * self.l_mask + self.eye
        u = self.u * self.u_mask + torch.diag(self.log_s.exp() * self.sign_s)
        return torch.matmul(self.p, torch.matmul(l, u))
        
    def inv_weight(self):
        l = self.l * self.l_mask + self.eye
        u = self.u * self.u_mask + torch.diag(self.log_s.exp() * self.sign_s)
        return torch.matmul(u.inverse(), torch.matmul(l.inverse(), self.p.inverse()))
    
    def forward(self, x):
        # x --> z
        logdet = self.log_s.sum() * x.shape[2] * x.shape[3]
        if self.train_sampling:
            # reverse path is used for training, take matrix inverse here
            weight = self.inv_weight()
            logdet = -logdet
        else:
            weight = self.weight()

        kernel = weight.view(*self.w_shape, 1, 1)
        return F.conv2d(x, kernel), logdet

    def reverse(self, x):
        # z --> x
        logdet = self.log_s.sum() * x.shape[2] * x.shape[3]
        if self.train_sampling:
            # reverse path is used for training, do not take inverse here
            weight = self.weight()
            logdet = -logdet
        else:
            weight = self.inv_weight()
        kernel = weight.view(*self.w_shape, 1, 1)
        return F.conv2d(x, kernel), logdet



class Conv2dZeros(nn.Module):
    """Normal conv2d for reparameterize the latent variable.
    - weight and bias initialized to zero
    - scale channel-wise after conv2d
    """
    def __init__(self, in_channels, out_channels):
        super(Conv2dZeros, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=True)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x * torch.exp(self.scale * 3)



class _CouplingNN(nn.Sequential):
    def __init__(self, in_features, out_features, width=128):
        # TODO: wide hidden layers, check wide resnet
        # TODO: I think the conv-norm-relu can be rearranged...
        super(_CouplingNN, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_features, width, 
            kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm1', ActNorm(width, return_logdet=False))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', nn.Conv2d(width, width, kernel_size=1, 
            stride=1, padding=0, bias=False))
        self.add_module('norm2', ActNorm(width, return_logdet=False))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv3', Conv2dZeros(width, out_features))



class _DenseCoupling(nn.Sequential):
    """ out_features = in_features
    """
    # previous K=16
    def __init__(self, in_features, out_features,
        num_layers=3, growth_rate=16, drop_rate=0.):
        super(_DenseCoupling, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
        # reduce the feature maps to in_features
        num_features = in_features + num_layers * growth_rate
        reduce = nn.Sequential(OrderedDict([
            ('norm1', nn.BatchNorm2d(num_features)),
            ('relu1', nn.ReLU()),
            ('conv_zero', Conv2dZeros(num_features, out_features))
            ]))
        self.add_module('reduce', reduce)


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer. Check Feistel cipher.

    Args:
        in_features (int): # features of input, before spliting to 2 parts
        cond_features (int): # features of conditioning
        coupling_net (str): choices=['dense', 'wide']
    """
    def __init__(self, in_features, cond_features, coupling_net='dense'):
        super(AffineCouplingLayer, self).__init__()
        # assert in_features % 2 == 0, '# input features must be evenly split,'\
        #     'but got {} features'.format(in_features)
        if in_features % 2 == 0:
            in_channels = in_features // 2 + cond_features
            out_channels = in_features
        else:
            # chunk is be (2, 1) if in_features==3
            in_channels = in_features // 2 + 1 + cond_features
            out_channels = in_features - 1
        if coupling_net == 'dense':
            coupling_nn = _DenseCoupling(in_channels, out_channels, 
                num_layers=3, growth_rate=16, drop_rate=0.)
        elif coupling_net == 'wide':
            coupling_nn = _CouplingNN(in_channels, out_channels, width=128)
        self.coupling_nn = coupling_nn

    def forward(self, x, cond):
        # cond: conditioning, for now, just concatenate
        # last chunk is smaller if not divided
        x1, x2 = x.chunk(2, 1)
        h = self.coupling_nn(torch.cat((x1, cond), 1))
        shift = h[:, 0::2]
        # scale = h[:, 1::2].exp()
        scale = torch.sigmoid(h[:, 1::2] + 2.)
        x2 = x2 + shift
        x2 = x2 * scale
        logdet = scale.log().view(x.shape[0], -1).sum(1)
        return torch.cat((x1, x2), 1), logdet

    def reverse(self, y, cond):
        y1, y2 = y.chunk(2, 1)
        h = self.coupling_nn(torch.cat((y1, cond), 1))
        shift = h[:, 0::2]
        scale = torch.sigmoid(h[:, 1::2] + 2.)
        y2 = y2 / scale
        y2 = y2 - shift
        logdet = scale.log().view(y.shape[0], -1).sum(1)
        return torch.cat((y1, y2), 1), logdet



class RevLayer(nn.Module):
    """Reversible layer, including:
        - normalization (actnorm)
        - conv 1x1 (learned permutation)
        - affine coupling layer
    """
    def __init__(self, in_features, cond_features, LUdecompose=False, 
        train_sampling=True, coupling_net='dense'):
        super(RevLayer, self).__init__()
        self.norm = ActNorm(in_features)
        if LUdecompose:
            self.conv1x1 = InvertibleConv1x1LU(in_features, 
                train_sampling=train_sampling)
        else:
            self.conv1x1 = InvertibleConv1x1(in_features,
                train_sampling=train_sampling)
        self.coupling = AffineCouplingLayer(in_features, cond_features, 
            coupling_net=coupling_net)

    def forward(self, x, cond):
        x, logdet1 = self.norm(x)
        x, logdet2 = self.conv1x1(x)
        x, logdet3 = self.coupling(x, cond)
        return x, logdet1 + logdet2 + logdet3

    def reverse(self, y, cond):
        y, logdet1 = self.coupling.reverse(y, cond)
        y, logdet2 = self.conv1x1.reverse(y)
        y, logdet3 = self.norm.reverse(y)
        return y, logdet1 + logdet2 + logdet3


class FirstRevLayer(nn.Module):
    """First reversible layer, only including:
        - affine coupling layer

    Note that features at the input scale is the conditioning.
    """
    def __init__(self, in_features, cond_features, coupling_net='dense'):
        super(FirstRevLayer, self).__init__()
        self.coupling = AffineCouplingLayer(in_features, cond_features, 
            coupling_net=coupling_net)

    def forward(self, x, cond):
        x, logdet = self.coupling(x, cond)
        return x, logdet

    def reverse(self, y, cond):
        y, logdet1 = self.coupling.reverse(y, cond)
        return y, logdet1


class Squeeze(nn.Module):
    """Squeeze the image from (B, C, H, W) to (B, C*4, H/2, W/2) for factor==2.
    """
    def __init__(self, factor=2):
        super(Squeeze, self).__init__()
        assert factor >= 1
        if factor == 1:
            Warning('Squeeze factor is 1, this is identity function')
        self.factor = factor

    def forward(self, x):
        if self.factor == 1:
            return x
        # n_channels, height, width
        C, H, W = x.shape[1:]
        assert H % self.factor == 0 and W % self.factor == 0
        x = x.reshape(-1, C, self.factor, H//self.factor, self.factor, W//self.factor)
        x = x.transpose(3, 4)
        x = x.reshape(-1, C * self.factor ** 2, H//self.factor, W//self.factor)
        return x

    def reverse(self, x):
        if self.factor == 1:
            return x
        C, H, W = x.shape[1:]
        assert C >= self.factor ** 2 and C % self.factor ** 2 == 0
        x = x.reshape(-1, C // self.factor ** 2, self.factor, self.factor, H, W)
        x = x.transpose(3, 4)
        x = x.reshape(-1, C // self.factor ** 2, H * self.factor, W * self.factor)
        return x


class GaussianDiag(object):
    Log2PI = float(np.log(2 * np.pi))

    def __init__(self, mean, log_stddev):
        super().__init__()
        self.mean = mean
        self.log_stddev = log_stddev.clamp_(min=-10., max=math.log(5.))
        # self._backward_hook = self.log_stddev.register_hook(
        #     lambda grad: torch.clamp_(grad, -10., 10.))

    def likelihood(self, x):
        like =  -0.5 * (GaussianDiag.Log2PI + self.log_stddev * 2. \
            + (x - self.mean) ** 2 / (self.log_stddev * 2.).exp())
        
        return like

    def log_prob(self, x):
        likelihood = self.likelihood(x)
        return likelihood.view(x.shape[0], -1).sum(1)

    def sample(self, eps=None):
        self.log_stddev.data.clamp_(min=-10., max=math.log(5.))
        if eps is None:
            eps = torch.randn_like(self.log_stddev)
        return self.mean + self.log_stddev.exp() * eps


class LatentEncoder(nn.Module):
    """The transform from the split z2 to mean and log_stddev of latent,
    which is assumed to be diagonal Gaussian in this case, similar to VAE.

    Simple encoder for intermediate latent variable in the flow part.
    """
    def __init__(self, in_channels):
        super(LatentEncoder, self).__init__()
        self.conv2d = Conv2dZeros(in_channels, in_channels * 2)

    def forward(self, x):
        mean, log_stddev = self.conv2d(x).chunk(2, 1)
        return GaussianDiag(mean, log_stddev)


class InputEncoder(nn.Sequential):
    """Encoder network for the input: x --> z
        output multiscale features after one pass

    Input --> DenseBlock (out) --> DownSampling --> DenseBlock (out) --> DownSampling --> ...
                DenseBlock (out) --> DownSampling --> DenseBlock (out)
    treat output of denseblock as the extracted features

    TODO: May need to control the output features at each scale.

    Args:
        in_channels (int): input features (x)
        latent_features (int): # features for top latent
    """
    # previous K = 12, init_features = 32
    def __init__(self, in_channels, latent_features, blocks, growth_rate=16,
                init_features=48, drop_rate=0.):
        super().__init__()
        # self.add_module('in_conv', nn.Conv2d(in_channels, init_features, 
        #                 kernel_size=7, stride=2, padding=3, bias=False))
        num_features = in_channels
        self.num_blocks = len(blocks)
        for i, num_layers in enumerate(blocks):
            if i == 0:
                block = _DenseBlockInput(num_layers=num_layers, 
                                        in_features=in_channels, 
                                        init_features=init_features, 
                                        growth_rate=growth_rate,
                                        drop_rate=drop_rate)
                num_features = init_features + (num_layers - 1) * growth_rate
                bottleneck = False
            else:
                block = _DenseBlock(num_layers=num_layers,
                                    in_features=num_features,
                                    growth_rate=growth_rate,
                                    drop_rate=drop_rate)
                num_features = num_features + num_layers * growth_rate
                bottleneck = True
            self.add_module('dense_block%d' % (i + 1), block)
            if i < len(blocks) - 1:
                trans_down = _Transition(in_features=num_features,
                                        out_features=num_features // 2,
                                        down=True, 
                                        drop_rate=drop_rate,
                                        bottleneck=bottleneck)
                self.add_module('trans_down%d' % (i + 1), trans_down)
                num_features = num_features // 2
        # used to parameterize the top latent z_top
        self.add_module('top_latent', Conv2dZeros(num_features, latent_features*2))

    def forward(self, x):
        conditions = []
        for i in range(self.num_blocks):
            # denseblock
            x = self[i*2](x)
            conditions.append(x)
            # downsampling, the last one is top_latent
            x = self[i*2+1](x)
            if i == self.num_blocks - 1:
                mean, log_stddev = x.chunk(2, 1)
                log_stddev = log_stddev.data.clamp_(min=-10., max=math.log(5.))
        return conditions, GaussianDiag(mean, log_stddev)

    def feature_sizes(self, x):
        # output feature sizes at different scales
        # x is test input
        out_sizes = []
        for i in range(self.num_blocks):
            # denseblock
            x = self[i*2](x)
            out_sizes.append(x.shape[1:])
            # downsampling
            if i < self.num_blocks - 1:
                x = self[i*2+1](x)
        if x.shape[1] % 2 == 0:
            Warning(f'coarse feature sizes is not even, got {x.shape}')
        return out_sizes


# good code -- one functionality within one function
class Split(nn.Module):
    """Factoring out half features after each RevBlock to reduce parameters and
    computation. This class only handles spliting in the intermediate layers.
    """

    def __init__(self, in_features):
        super(Split, self).__init__()
        self.latent_encoder = LatentEncoder(in_features // 2)

    def forward(self, z, return_eps=False):
        # split out z2, and evalute log prob at z2 which takes the form of 
        # diagonal Gaussian are reparameterized by latent_encoder
        z, z2 = z.chunk(2, 1)
        prior = self.latent_encoder(z)
        log_prob_prior = prior.log_prob(z2)
        if return_eps:
            eps = (z2 - prior.mean) / prior.log_stddev.exp()
        else:
            eps = None
        return z, log_prob_prior, eps

    def reverse(self, z1, eps=None):
        # sample z2, then concat with z1
        # intermediate flow, z2 is the split-out latent
        prior = self.latent_encoder(z1)
        z2 = prior.sample(eps)
        z = torch.cat((z1, z2), 1)
        log_prob_prior = prior.log_prob(z2)
        return z, log_prob_prior


class RevBlock(nn.Module):
    """Reversible block, contains:
    - `Squeeze`
    - A cascade of `RevLayer`s
    - `Split` (no Split for the top latent)

    Args:
        in_features (int): # input feature maps
        n_layers (int): # `RevLayer`s in this block
    """
    def __init__(self, in_features, cond_features, n_layers, coupling_net='dense',
        factor=2, LUdecompose=False, train_sampling=True, do_split=True):
        super(RevBlock, self).__init__()
        self.do_split = do_split
        self.squeeze = Squeeze(factor)
        in_features = in_features * factor ** 2
        self.revlayers = nn.Sequential()
        for i in range(n_layers):
            self.revlayers.add_module('revlayer{}'.format(i+1), 
                RevLayer(in_features, cond_features, LUdecompose=LUdecompose,
                train_sampling=train_sampling, coupling_net=coupling_net))
        if do_split:
            self.split = Split(in_features)

    def forward(self, x, cond, return_eps=False):
        logdet = 0.
        x = self.squeeze(x)
        for revlayer in self.revlayers._modules.values():
            # conditioning enters at every revlayer
            x, dlogdet = revlayer(x, cond)
            logdet = logdet + dlogdet
        if self.do_split:
            x, log_prob_prior, eps = self.split(x, return_eps=return_eps)
            logdet = logdet + log_prob_prior
            return x, logdet, eps
        else:
            return x, logdet, None

    def reverse(self, y, cond, eps):
        # eps is not used if this revblock does not split in the end
        # focus on sampling
        logdet = 0.
        if self.do_split:
            y, log_prob_prior = self.split.reverse(y, eps)
            logdet = logdet + log_prob_prior
        for revlayer in reversed(self.revlayers._modules.values()):
            y, dlogdet = revlayer.reverse(y, cond)
            logdet = logdet + dlogdet
        return self.squeeze.reverse(y), logdet
 

class FirstRevBlock(nn.Module):
    """First Reversible block, contains:
        - A cascade of `RevLayer`s, where the first RevLayer does not include
            ActNorm and 1x1 conv

    Args:
        in_features (int): # input feature maps
        n_layers (int): # `RevLayer`s in this block
    """
    def __init__(self, in_features, cond_features, n_layers, coupling_net='dense',
        LUdecompose=False, train_sampling=True):
        super(FirstRevBlock, self).__init__()
        self.revlayers = nn.Sequential()
        self.revlayers.add_module('revlayer1',  
            FirstRevLayer(in_features, cond_features))
        for i in range(1, n_layers):
            self.revlayers.add_module('revlayer{}'.format(i+1), 
                RevLayer(in_features, cond_features, LUdecompose=LUdecompose,
                train_sampling=train_sampling, coupling_net=coupling_net))

    def forward(self, x, cond):
        logdet = 0.
        for revlayer in self.revlayers._modules.values():
            x, dlogdet = revlayer(x, cond)
            logdet = logdet + dlogdet
        return x, logdet

    def reverse(self, y, cond):
        # focus on sampling
        logdet = 0.
        for revlayer in reversed(self.revlayers._modules.values()):
            y, dlogdet = revlayer.reverse(y, cond)
            logdet = logdet + dlogdet
        return y, logdet


class MultiScaleCondGlow(nn.Module):
    """Multiscale conditional Glow

    Default: split features after each RevLayer except the first and last layer.

    Args:
        img_size(int):
        x_channels (int): # Channels of input images (conditioning)
        y_channels (int): # channels of output images (prediction)
        enc_blocks (list of int): encoder blocks x --> z
        flow_blocks (list of int): Block configurations, how many `RevLayer`s 
            in each `RevBlock`
        squeeze_factor (int): Squeeze factor for `Squeeze`
        LUdecompose (bool): use LU decompose for matrix inverse in 1x1 conv
        train_sampling (bool): use sampling as the main path for training, e.g.
            reverse KL setting; if MLE, use False
        data_init (bool, False): data initalization for ActNorm
    """

    def __init__(self, img_size, x_channels, y_channels, enc_blocks, 
        flow_blocks, flow_coupling='dense', squeeze_factor=2, LUdecompose=False, 
        train_sampling=True, data_init=False):
        super().__init__()
        if isinstance(img_size, int):
            self.img_size = [img_size, img_size]
        elif isinstance(img_size, (list, tuple)):
            assert len(img_size) == 2, 'Images, 2D!'
            self.img_size = list(img_size)
        # data initialized
        self.data_init = data_init
        self.data_initialized = False
        self.y_channels = y_channels        
        self.flow_blocks = flow_blocks
        self.factor = squeeze_factor
        # get the top latent channels
        z_shapes = self._z_shapes()
        top_features = z_shapes[-1][0]
        # input encoder, extract features at multiscales
        self.encoder = InputEncoder(x_channels, top_features, enc_blocks, 
            growth_rate=16, init_features=48, drop_rate=0.)
        
        test_x = torch.randn(1, x_channels, img_size, img_size)
        cond_sizes = self.encoder.feature_sizes(test_x)

        # y <---> (z, x_cond)
        self.flow = nn.Sequential()
        n_features = y_channels
        split = True
        for i, n_layers in enumerate(flow_blocks): 
            if i == 0:
                revblock = FirstRevBlock(n_features, cond_sizes[i][0], n_layers,
                    coupling_net=flow_coupling, LUdecompose=LUdecompose, 
                    train_sampling=train_sampling)
            else:
                if i == len(flow_blocks) - 1:
                    # top latent, no split (from encoder)
                    split = False
                revblock = RevBlock(n_features, cond_sizes[i][0], n_layers, 
                    coupling_net=flow_coupling, factor=squeeze_factor, 
                    LUdecompose=LUdecompose, train_sampling=train_sampling, 
                    do_split=split)
                n_features = n_features * (squeeze_factor ** 2) // 2
            self.flow.add_module('revblock{}'.format(i+1), revblock)
        
        if self.data_init:
            for name, module in self.named_modules():
                if isinstance(module, ActNorm):
                    module.data_init = True
  
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def model_size(self):
        return module_size(self)

    def forward(self, y, x, return_eps=False):
        """ p(y|x) where (x, y) are pair of input & output
        y --> z, evaluate det(dz/dy) and p(z|x) --> p(y|x)

        Args:
            y (Tensor): output
            x (Tensor): input

        Returns:
            z, logp(y|x), eps_list (None if return_eps is False)
        """
        logdet = 0.
        # list of conditioning features at different scales, and conditional prior
        conditions, cond_prior = self.encoder(x)
        eps_list = []
        for i, module in enumerate(self.flow._modules.values()):
            if i == 0:
                # first revblock, no squeeze and split
                y, dlogdet = module(y, conditions[i])
            elif i == len(self.flow_blocks) - 1:
                # last revblock, top latent
                y, dlogdet, _ = module(y, conditions[i])
                log_prior = cond_prior.log_prob(y)
                if return_eps:
                    eps = (y - cond_prior.mean) / cond_prior.log_stddev.exp()
                    eps_list.append(eps)
                logdet = logdet + log_prior
            else:
                # middel revblocks, squeeze and split latent
                y, dlogdet, eps = module(y, conditions[i], return_eps=return_eps)
                if return_eps:
                    eps_list.append(eps)
            logdet = logdet + dlogdet
        # y is actually z, latent
        if return_eps:
            return y, logdet, eps_list
        else:
            return y, logdet, None


    def generate(self, x, eps_list=None):
        """Given input x, generate samples from p(y|x), One sample y for each x.
        This function is used during training.

        Args:
            x (Tensor): batch of input
            eps_list (list of 4D Tensors): include all the eps for sampling 
                latent variables, to check the shapes of each latent variable,
                refer to `Glow._z_shapes()`
        Returns:
            y, logp
            y: output
            logp: logp(y|x)
        """
        if eps_list is not None:
            assert len(eps_list) == len(self.flow_blocks)-1, 'The specified noise '\
                'must have the same size as the latent variables'
        else:
            eps_list = [None] * (len(self.flow_blocks) - 1)
        # the first [None] is for the first RevBlock, where there is no latent
        eps_list = [None] + eps_list
        logp = 0.
        conditions, cond_prior = self.encoder(x)
        # sample z from cond prior p(z|x), one z for one x
        # the top latent is sampled from the cond prior built with InputEncoder
        z = cond_prior.sample(eps_list[-1])
        log_prob_prior = cond_prior.log_prob(z)
        logp = logp + log_prob_prior

        # sample y from flow-network z --> y
        # eps_list[-1] is not useful for RevBlock inverse
        for i, ((name, module), cond, eps) in enumerate(
            zip(reversed(self.flow._modules.items()), 
            reversed(conditions), reversed(eps_list))):
            if i == len(self.flow_blocks) - 1:
                z, logdet = module.reverse(z, cond)
            else:
                z, logdet = module.reverse(z, cond, eps)
            logp = logp + logdet
        return z, logp


    def approx_pred_mean(self, x):
        """Using cheap approximation to compute the predictive mean.
        For each Gaussian in the model, just use its mean to compute the output.
        """
        eps_zeros = self.create_zero_noise(batch_size=x.shape[0])
        y, logp = self.generate(x, eps_list=eps_zeros)
        return y, logp


    def sample(self, x, n_samples, eps_list=None, temperature=None):
        """Given batch input x, generate `n_samples` of y from p(y|x) for each 
        x in the batch. Should provide eps_list if visualization is needed.

        Args:
            x (Tensor): (B, xC, xH, xW)
            eps_list (Tensor): if not None, size (n_samples, B, zC, zH, zW)
        Returns:
            samples (Tensor): (n_samples, B, yC, yH, yW)
        """
        if temperature is None:
            temperature = 0.7
        if eps_list is not None:
            assert n_samples == eps_list[-1].shape[0] and \
                 x.shape[0] == eps_list[-1].shape[1]
        else:
            eps_list = self.create_fixed_noise(n_samples, 
                batch_size=x.shape[0])
        eps_list = [None] + eps_list
        conditions, cond_prior = self.encoder(x)
        y_list = []
        for i in range(n_samples):
            # generate one sample for each x
            z = cond_prior.sample(eps_list[-1][i])
            # flow reverse
            for module, cond, eps in zip(reversed(self.flow._modules.values()), 
                                   reversed(conditions), reversed(eps_list)):
                if eps is None:
                    z, _ = module.reverse(z, cond)
                else:
                    z, _ = module.reverse(z, cond, eps[i] * temperature)
            y_list.append(z)
        return torch.stack(y_list, 0)

    def _z_shapes(self):
        """Computes the shapes of latent variables which are distributed in 
        the end of each RevBlock.

        Args:
            img_size (int or list of size 2): assumes images
        """
        feature_size = self.img_size
        n_features = self.y_channels
        z_shapes = []
        for _ in range(len(self.flow_blocks) - 2):
            feature_size = [fs // 2 for fs in feature_size]
            n_features = n_features * self.factor ** 2 // 2
            z_shapes.append((n_features, *feature_size))
        # top latent does not factor out
        feature_size = [fs // 2 for fs in feature_size]
        z_shapes.append((n_features * self.factor ** 2 , *feature_size))
        return z_shapes

    def create_fixed_noise(self, n_samples, batch_size=1):
        """Create fixed noise for visualizing the training process, 
        for one input x. To sample multiple, run multiple times.

        Args:
            n_samples (int):
        """
        eps_list = []
        z_shapes = self._z_shapes()
        for z_shape in z_shapes:
            eps_list.append(torch.randn(n_samples, batch_size, *z_shape).to(self.device))
        return eps_list

    def create_zero_noise(self, batch_size):
        """Create fixed noise for visualizing the training process, 
        for one input x. To sample multiple, run multiple times.

        Args:
            n_samples (int):
        """
        eps_list = []
        z_shapes = self._z_shapes()
        for z_shape in z_shapes:
            eps_list.append(torch.zeros(batch_size, *z_shape).to(self.device))
        return eps_list

    def init_actnorm(self):
        for name, module in self.named_modules():
            if isinstance(module, ActNorm):
                module.data_initialized = True
        self.data_initialized = True

    def predict(self, x_test, n_samples=20, temperature=1.0):
        """
        Given `x_test`, return pred mean and var.

        Args:
            x_test (Tensor): (N, *)
            n_samples (int): # samples to estimate the mean and var
        Returns:
            pred_mean, pred_var
        """
        # (n_samples, N, *)
        pred = self.sample(x_test, n_samples, temperature=temperature)
        return pred.mean(0), pred.var(0)
    
    def propagate(self, mc_loader, n_samples=20, temperature=1.0, var_samples=10):
        """Uncertainty propagation
        E[Y] = E_X E[Y|X]
        Var[Y] = E_X Var(Y|X) + Var_X E[Y|X]
        """
        # S x oC x oH x oW
        output_size = mc_loader.dataset[0][1].shape
        Ey = torch.zeros(var_samples, *output_size, device=self.device)
        Eyy, Vy = torch.zeros_like(Ey), torch.zeros_like(Ey)

        for i in range(var_samples):
            print(f'propagating for the {i}-th time...')
            # repeat approximation of mean and var for `var_samples` times
            for _, (x_mc, _) in enumerate(mc_loader):
                # (S, B, C, H, W)
                x_mc = x_mc.to(self.device)
                y = self.sample(x_mc, n_samples=n_samples, temperature=temperature)
                # (B, C, H, W)
                y_mean = y.mean(0)
                y2_mean = y.pow(2).mean(0)
                # compute mean over B in mini-batch, (C, H, W)
                Ey[i] += y_mean.mean(0)
                Eyy[i] += y2_mean.mean(0)

        Ey /= len(mc_loader)
        Eyy /= len(mc_loader)
        Vy = Eyy - Ey ** 2

        # compute statistics of statistics
        return Ey.mean(0), Ey.var(0), Vy.mean(0), Vy.var(0)
