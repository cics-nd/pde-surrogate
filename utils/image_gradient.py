"""
Estimating image gradient with convolution.
Potential to extend to higher order terms.

Reference:
Sobel filter
https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/edges.py

Fourier filter
    http://www.cns.nyu.edu/pub/lcv/farid03-reprint.pdf

Low Degree Chebyshev (LDC) differentiation
this is a global method, seems not to be effective...
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import sys
from torch.nn.modules.utils import _quadruple


class SobelFilter(object):

    def __init__(self, imsize, correct=True, device='cpu'):
        # conv2d is cross-correlation, need to transpose the kernel here
        self.HSOBEL_WEIGHTS_3x3 = torch.FloatTensor(
            np.array([[-1, -2, -1],
                     [ 0, 0, 0],
                     [1, 2, 1]]) / 8.0).unsqueeze(0).unsqueeze(0).to(device)

        self.VSOBEL_WEIGHTS_3x3 = self.HSOBEL_WEIGHTS_3x3.transpose(-1, -2)

        self.VSOBEL_WEIGHTS_5x5 = torch.FloatTensor(
                    np.array([[-5, -4, 0, 4, 5],
                                [-8, -10, 0, 10, 8],
                                [-10, -20, 0, 20, 10],
                                [-8, -10, 0, 10, 8],
                                [-5, -4, 0, 4, 5]]) / 240.).unsqueeze(0).unsqueeze(0).to(device)
        self.HSOBEL_WEIGHTS_5x5 = self.VSOBEL_WEIGHTS_5x5.transpose(-1, -2)

        modifier = np.eye(imsize)
        modifier[0:2, 0] = np.array([4, -1])
        modifier[-2:, -1] = np.array([-1, 4])
        self.modifier = torch.FloatTensor(modifier).to(device)
        self.correct = correct


    def grad_h(self, image, filter_size=3):
        """Get image gradient along horizontal direction, or x axis.
        Option to do replicate padding for image before convolution. This is mainly
        for estimate the du/dy, enforcing Neumann boundary condition.

        Args:
            image (Tensor): (1, 1, H, W)
            replicate_pad (None, int, 4-tuple): if 4-tuple, (padLeft, padRight, padTop, 
                padBottom)
        """
        image_width = image.shape[-1]

        if filter_size == 3:
            replicate_pad = 1
            kernel = self.VSOBEL_WEIGHTS_3x3
        elif filter_size == 5:
            replicate_pad = 2
            kernel = self.VSOBEL_WEIGHTS_5x5
        image = F.pad(image, _quadruple(replicate_pad), mode='replicate')
        grad = F.conv2d(image, kernel, stride=1, padding=0, bias=None) * image_width
        # modify the boundary based on forward & backward finite difference (three points)
        # forward [-3, 4, -1], backward [3, -4, 1]
        if self.correct:
            return torch.matmul(grad, self.modifier)
        else:
            return grad

    def grad_v(self, image, filter_size=3):
        image_height = image.shape[-2]
        if filter_size == 3:
            replicate_pad = 1
            kernel = self.HSOBEL_WEIGHTS_3x3
        elif filter_size == 5:
            replicate_pad = 2
            kernel = self.HSOBEL_WEIGHTS_5x5
        image = F.pad(image, _quadruple(replicate_pad), mode='replicate')
        grad = F.conv2d(image, kernel, stride=1, padding=0, 
            bias=None) * image_height
        # modify the boundary based on forward & backward finite difference
        if self.correct:
            return torch.matmul(self.modifier.t(), grad)
        else:
            return grad


def gaussian_filter1d_weights(sigma, order=0, truncate=4.0):
    """One-dimensional Gaussian filter.
    https://github.com/scipy/scipy/blob/v0.16.1/scipy/ndimage/filters.py#L181

    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : {0, 1, 2, 3}, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. An order of 1, 2, or 3 corresponds to convolution with
        the first, second or third derivatives of a Gaussian. Higher
        order derivatives are not implemented
    %(output)s
    %(mode)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    Returns
    -------
    gaussian_filter1d : ndarray
    """
    if order not in range(4):
        raise ValueError('Order outside 0..3 not implemented')
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    # implement first, second and third order derivatives:
    if order == 1:  # first derivative
        weights[lw] = 0.0
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = -x / sd * weights[lw + ii]
            weights[lw + ii] = -tmp
            weights[lw - ii] = tmp
    elif order == 2:  # second derivative
        weights[lw] *= -1.0 / sd
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = (x * x / sd - 1.0) * weights[lw + ii] / sd
            weights[lw + ii] = tmp
            weights[lw - ii] = tmp
    elif order == 3:  # third derivative
        weights[lw] = 0.0
        sd2 = sd * sd
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = (3.0 - x * x / sd) * x * weights[lw + ii] / sd2
            weights[lw + ii] = -tmp
            weights[lw - ii] = tmp

    return np.array(weights)
    

class GaussianFilter(object):
    """Gaussian smoothing

    Only use `reflect` mode for padding
    """
    def __init__(self, sigma=1.0, truncate=4.0, order=0, device='cpu'):

        gaussian_weights_1d = gaussian_filter1d_weights(sigma, 
            order=order, truncate=truncate)
        weights = np.expand_dims(gaussian_weights_1d, 1)

        self.weights = torch.FloatTensor(
            np.matmul(weights, weights.T)).unsqueeze(0).unsqueeze(0).to(device)

    def __call__(self, image):
        # image: (B, C, H, W)
        padding = (self.weights.shape[-1] - 1) // 2
        image = F.pad(image, _quadruple(padding), mode='reflect')
        channels = image.shape[1]
        weights = self.weights.repeat(channels, 1, 1, 1)
        return F.conv2d(image, weights, bias=None, stride=1, padding=0, groups=channels)


# class SobelFilterV2(nn.Module):

#     def __init__(self, device='cpu'):
#         self.HSOBEL_WEIGHTS_3x3 = torch.FloatTensor(
#             np.array([[1, 2, 1],
#                      [ 0, 0, 0],
#                      [-1,-2,-1]]) / 8.0).unsqueeze(0).unsqueeze(0).to(device)

#         self.VSOBEL_WEIGHTS_3x3 = self.HSOBEL_WEIGHTS_3x3.transpose(-1, -2)

#         self.VSOBEL_WEIGHTS_5x5 = torch.FloatTensor(
#                     np.array([[5, 4, 0, -4, -5],
#                                 [8, 10, 0, -10, -8],
#                                 [10, 20, 0, -20, -10],
#                                 [8, 10, 0, -10, -8],
#                                 [5, 4, 0, -4, -5]]) / 240.).unsqueeze(0).unsqueeze(0).to(device)
#         self.HSOBEL_WEIGHTS_5x5 = self.VSOBEL_WEIGHTS_5x5.transpose(-1, -2)


#     def grad_h(self, image, filter_size=3):
#         """Get image gradient along horizontal direction, or x axis.
#         Option to do replicate padding for image before convolution. This is mainly
#         for estimate the du/dy, enforcing Neumann boundary condition.

#         Args:
#             image (Tensor): (1, 1, H, W)
#             replicate_pad (None, int, 4-tuple): if 4-tuple, (padLeft, padRight, padTop, 
#                 padBottom)
#         """
#         image_width = image.shape[-1]
#         if filter_size == 3:
#             replicate_pad = 1
#             kernel = self.VSOBEL_WEIGHTS_3x3
#         elif filter_size == 5:
#             replicate_pad = 2
#             kernel = self.VSOBEL_WEIGHTS_5x5
#         image = F.pad(image, _quadruple(replicate_pad), mode='replicate')
#         return F.conv2d(image, kernel, stride=1, padding=0, 
#             bias=None) * image_width


#     def grad_v(self, image, filter_size=5):
#         image_height = image.shape[-2]
#         if filter_size == 3:
#             replicate_pad = 1
#             kernel = self.HSOBEL_WEIGHTS_3x3
#         elif filter_size == 5:
#             replicate_pad = 2
#             kernel = self.HSOBEL_WEIGHTS_5x5
#         image = F.pad(image, _quadruple(replicate_pad), mode='replicate')
#         return F.conv2d(image, kernel, stride=1, padding=0, 
#             bias=None) * image_height


class FourierFilter(object):
    """New derivative filter
    http://www.cns.nyu.edu/pub/lcv/farid03-reprint.pdf
    """
    def __init__(self, device='cpu'):
        # TODO maybe with higher precision
        p3 = np.array([0.229879, 0.540242, 0.229879])
        d1_3 = np.array([-0.425287, 0., 0.425287])
        p5 = np.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659])
        d1_5 = np.array([-0.109604, -0.276691, 0.00000, 0.276691, 0.109604])
        p7 = np.array([0.005412, 0.069591, 0.244560, 0.360875, 0.244560, 0.069591, 0.005412])
        d1_7 = np.array([-0.019479, -0.123915, -0.193555, 0.000000, 0.193555, 0.123915, 0.019479])

        self.kernel_h_3x3 = torch.FloatTensor(
            p3[None, :].T @ d1_3[None, :]).unsqueeze(0).unsqueeze(0).to(device)

        self.kernel_h_5x5 = torch.FloatTensor(
            p5[None, :].T @ d1_5[None, :]).unsqueeze(0).unsqueeze(0).to(device)

        self.kernel_h_7x7 = torch.FloatTensor(
            p7[None, :].T @ d1_7[None, :]).unsqueeze(0).unsqueeze(0).to(device)

    def grad_h(self, image, filter_size=5):
        # horizontal derivative
        image_width = image.shape[-1]
        if filter_size == 3:
            replicate_pad = 1
            kernel = self.kernel_h_3x3
        elif filter_size == 5:
            replicate_pad = 2
            kernel = self.kernel_h_5x5
        elif filter_size == 7:
            replicate_pad = 3
            kernel = self.kernel_h_7x7
        image = F.pad(image, _quadruple(replicate_pad), mode='replicate')
        return F.conv2d(image, kernel, stride=1, padding=0, 
            bias=None) * image_width

    def grad_v(self, image, filter_size=5):
        # vertical derivative
        image_height = image.shape[-2]
        if filter_size == 3:
            replicate_pad = 1
            kernel = self.kernel_h_3x3.transpose(-1, -2)
        elif filter_size == 5:
            replicate_pad = 2
            kernel = self.kernel_h_5x5.transpose(-1, -2)
        elif filter_size == 7:
            replicate_pad = 3
            kernel = self.kernel_h_7x7.transpose(-1, -2)
        image = F.pad(image, _quadruple(replicate_pad), mode='replicate')
        return F.conv2d(image, kernel, stride=1, padding=0, 
            bias=None) * image_height


def _mask_filter_result(result, mask):
    """Return result after masking.
    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is None:
        result[0, :] = 0
        result[-1, :] = 0
        result[:, 0] = 0
        result[:, -1] = 0
        return result


if __name__ == '__main__':

    from yinhao.utils.plot import plot_row
    data_dir = '/scratch365/yzhu10/data/grf_exp/ls0.1_ng65_inverse/verify'
    kle = 512
    fig_dir = data_dir + f'/figs_k{kle}_fwd_bwd'
    # fig_dir = data_dir + f'/figs_k{kle}'
    from yinhao.utils.misc import mkdirs
    mkdirs(fig_dir)
    idx = 7

    sigma = 0.25
    truncate = 4.0
    gaussian_filter = GaussianFilter(sigma=sigma, truncate=truncate)
    ks = gaussian_filter.weights.shape[-1]
    print(gaussian_filter.weights)
    print(gaussian_filter.weights.shape)

    v_x = np.loadtxt(data_dir + f'/output_k{kle}/{idx}_sigma_1.dat')

    v_x_smoothed = gaussian_filter(torch.FloatTensor(v_x).unsqueeze(0).unsqueeze(0))
    v_x_smoothed = v_x_smoothed.numpy()[0, 0]

    plot_row([v_x, v_x_smoothed, (v_x - v_x_smoothed)], fig_dir, f'v_x_idx{idx}_smooth_sigma{sigma}_trun{ks}', cmap='jet')

    sys.exit(0)

    filtering = 'sobel'

    K_arr = np.loadtxt(data_dir + f'/input_k{kle}/{idx}.dat')
    # K = torch.FloatTensor(K_arr).unsqueeze(0).unsqueeze(0)
    u_arr = np.loadtxt(data_dir + f'/output_k{kle}/{idx}_u.dat')
    u = torch.FloatTensor(u_arr).unsqueeze(0).unsqueeze(0)
    v_x = np.loadtxt(data_dir + f'/output_k{kle}/{idx}_sigma_1.dat')
    v_y = np.loadtxt(data_dir + f'/output_k{kle}/{idx}_sigma_2.dat')

    modifier = np.eye(K_arr.shape[-1])
    modifier[0:2, 0] = np.array([4, -1])
    modifier[-2:, -1] = np.array([-1, 4])

    if filtering == 'fourier':
        filter = FourierFilter()
        filter_sizes = [3, 5, 7]
    elif filtering == 'sobel':
        filter = SobelFilter(K_arr.shape[-1], correct=True)
        filter_sizes = [3, 5]

    for filter_size in filter_sizes:
        print(filter_size)
        u_x = filter.grad_h(u, filter_size)
        u_y = filter.grad_v(u, filter_size)

        u_x = u_x.detach().numpy()[0, 0]
        u_y = u_y.detach().numpy()[0, 0]

        # u_x = u_x @ modifier
        # u_y = modifier.T @ u_y

        v_x_est = -K_arr * u_x
        v_y_est = -K_arr * u_y
        
        # print(v_y[0], v_y[-1])
        # print(v_y_est[0], v_y_est[-1])

        plot_row([K_arr, u_arr, v_x, v_x_est, (v_x - v_x_est)], fig_dir, f'v_x_idx{idx}_{filtering}_{filter_size}', cmap='jet')
        plot_row([K_arr, u_arr, v_y, v_y_est, (v_y - v_y_est)], fig_dir, f'v_y_idx{idx}_{filtering}_{filter_size}', cmap='jet')
