"""
Darcy flow problem
8 cases...
primal/mixed + fc/conv + variational/residual
"""
import torch
import torch.autograd as ag
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def grad(outputs, inputs):
    return ag.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), 
                   create_graph=True)


def bilinear_interpolate_torch(im, x, y):
    # https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        dtype_long = torch.LongTensor
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)
    
    Ia = im[ y0, x0 ][0]
    Ib = im[ y1, x0 ][0]
    Ic = im[ y0, x1 ][0]
    Id = im[ y1, x1 ][0]
    
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))

    return torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) \
        + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)


def primal_residual_fc(model, x, K_grad_ver, K_grad_hor, K, verbose=False):
    """Computes the residules of satisifying PDE at x
    Permeability is also provided: K

    First assume x is on grid

    Args:
        model (Module): u = f(x) pressure network, input is spatial coordinate
        x (Tensor): (N, 2) spatial input x. could be off-grid, vary very pass
        grad_K (Tensor): estimated gradient field
        verbose (bool): If True, print info
    Returns:
        residual: (N, 1)
    """
 
    assert len(K_grad_ver) == len(x)
    x.requires_grad = True
    u = model(x)
    # grad outputs a tuple: (N, 2)
    u_x = grad(u, x)[0]
    
    div1 = K_grad_ver * u_x[:, 0] + K * grad(u_x[:, 0], x)[0][:, 0]
    div2 = K_grad_hor * u_x[:, 1] + K * grad(u_x[:, 1], x)[0][:, 1]
    div = div1 + div2

    if verbose:
        print(div.detach().mean(), div.detach().max(), div.detach().min())
    return (div ** 2).mean()

def neumann_boundary(model, x):
    # bug: u_y! NOT u_x
    x.requires_grad = True
    u = model(x)
    u_ver = grad(u, x)[0][:, 0]
    return (u_ver ** 2).mean()


def neumann_boundary_mixed(model, x):

    # x.requires_grad = True
    y = model(x)
    tau_ver = y[:, 1]
    
    return (tau_ver ** 2).mean()


def primal_variational_fc(model, x, K, verbose=False):
    """Evaulate energy functional. Simple MC. Evaluate on [1:-1, 1:-1] of grid

    Args:
        x (Tensor): colloc points on interior of grid (63 ** 2, 2)
    """
    x.requires_grad = True
    u = model(x)
    u_x = grad(u, x)[0]
    u_x_squared = (u_x ** 2).sum(1)
    energy = (0.5 * K * u_x_squared).mean()
    if verbose:
        print(f'energy: {energy:.6f}')
    return energy


def mixed_residual_fc(model, x, K, verbose=False, rand_colloc=False, fig_dir=None):
    """
    Args:
        x: (N, 2)
        K: (N, 1)

    """
    x.requires_grad = True
    # (N, 3)
    y = model(x)
    u = y[:, 0]
    # (N, 2)
    tau = y[:, [1, 2]]
    # (N, 2)
    u_x = grad(u, x)[0]

    grad_tau_ver = grad(y[:, 1], x)[0][:, 0]
    grad_tau_hor = grad(y[:, 2], x)[0][:, 1]

    if rand_colloc:
        K = bilinear_interpolate_torch(K.unsqueeze(-1), x[:, [1]], x[:, [0]])
        K = K.t()
        # print(f'K interp: {K.shape}')
        # plt.imshow(K[0].detach().cpu().numpy().reshape(65, 65))
        # plt.savefig(fig_dir+'/Kinterp.png')
        # plt.close()


    loss_constitutive = ((K * u_x + tau) ** 2).mean()
    loss_continuity = ((grad_tau_ver + grad_tau_hor) ** 2).mean()

    return loss_constitutive + loss_continuity


"""
ConvNet ============================================
"""

def energy_functional_exp(input, output, sobel_filter):
    r""" sigma = -exp(K * u) * grad(u)

    V(u, K) = \int 0.5 * exp(K*u) * |grad(u)|^2 dx
    """
    grad_h = sobel_filter.grad_h(output)
    grad_v = sobel_filter.grad_v(output)

    return (0.5 * torch.exp(input * output) * (grad_h ** 2 + grad_v ** 2)).mean()


def conv_constitutive_constraint(input, output, sobel_filter):
    """sigma = - K * grad(u)

    Args:
        input (Tensor): (1, 1, 65, 65)
        output (Tensor): (1, 3, 65, 65), 
            three channels from 0-2: u, sigma_1, sigma_2
    """
    grad_h = sobel_filter.grad_h(output[:, [0]])
    grad_v = sobel_filter.grad_v(output[:, [0]])
    est_sigma1 = - input * grad_h
    est_sigma2 = - input * grad_v

    return ((output[:, [1]] - est_sigma1) ** 2 
        + (output[:, [2]] - est_sigma2) ** 2).mean()


def conv_constitutive_constraint_nonlinear(input, output, sobel_filter, beta1, beta2):
    """Nonlinear extension of Darcy's law
        - K * grad_u = sigma + beta1 * sqrt(K) * sigma ** 2 + beta2 * K * sigma ** 3

    Args:
        input: K
        output: u, sigma1, sigma2
    """
    K_u_h = - input * sobel_filter.grad_h(output[:, [0]])
    K_u_v = - input * sobel_filter.grad_v(output[:, [0]])
    sigma = output[:, [1, 2]]
    rhs = sigma + beta1 * torch.sqrt(input) * sigma ** 2 + beta2 * input * sigma ** 3
    return ((K_u_h - rhs[:, [0]])** 2 + (K_u_v - rhs[:, [1]]) ** 2).mean()

def conv_constitutive_constraint_nonlinear_exp(input, output, sobel_filter):
    """Nonlinear extension of Darcy's law
        sigma = - exp(K * u) grad(u)

    Args:
        input: K
        output: u, sigma1, sigma2
    """
    grad_h = sobel_filter.grad_h(output[:, [0]])
    grad_v = sobel_filter.grad_v(output[:, [0]])

    sigma_h = - torch.exp(input * output[:, [0]]) * grad_h
    sigma_v = - torch.exp(input * output[:, [0]]) * grad_v

    return ((output[:, [1]] - sigma_h) ** 2 
        + (output[:, [2]] - sigma_v) ** 2).mean()

def conv_continuity_constraint(output, sobel_filter, use_tb=True):
    """
    div(sigma) = -f

    Args:

    """
    sigma1_x1 = sobel_filter.grad_h(output[:, [1]])
    sigma2_x2 = sobel_filter.grad_v(output[:, [2]])
    # leave the top and bottom row free, since sigma2_x2 is almost 0,
    # don't want to enforce sigma1_x1 to be also zero.
    if use_tb:
        return ((sigma1_x1 + sigma2_x2) ** 2).mean()
    else:
        return ((sigma1_x1 + sigma2_x2) ** 2)[:, :, 1:-1, :].mean()

def conv_boundary_condition(output):
    left_bound, right_bound = output[:, 0, :, 0], output[:, 0, :, -1]
    top_down_flux = output[:, 2, [0, -1], :]
    loss_dirichlet = F.mse_loss(left_bound, torch.ones_like(left_bound)) \
        + F.mse_loss(right_bound, torch.zeros_like(right_bound))
    loss_neumann = F.mse_loss(top_down_flux, torch.zeros_like(top_down_flux))

    return loss_dirichlet, loss_neumann
