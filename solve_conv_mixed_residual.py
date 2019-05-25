"""Solving Darcy Flow using ConvNet with mixed residual loss

Flow through Porous Media, 2D
div (K(s) grad u(s)) = 0, s = (s1, s2) in (0, 1) x (0, 1)
Boundary:   
    u = 1, s1 = 0; u = 0, s1 = 1
    u_s2 = 0, s2 in {0, 1}

Optimizer: L-BFGS    
Considered nonlinear PDE. (nonlinear corrections to Darcy)
"""

import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F
import torch.optim as optim

from models.codec import Decoder
from utils.image_gradient import SobelFilter
from models.darcy import conv_continuity_constraint as continuity_constraint
from models.darcy import conv_boundary_condition as boundary_condition
from utils.plot import save_stats, plot_prediction_det, plot_prediction_det_animate2
from utils.misc import mkdirs, to_numpy
import numpy as np
import argparse
import h5py
import sys
import time
import os
from pprint import pprint
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def main():
    parser = argparse.ArgumentParser(description='CNN to solve PDE')
    parser.add_argument('--exp-dir', type=str, default='./experiments/solver', help='color map')
    parser.add_argument('--nonlinear', action='store_true', default=False, help='set True for nonlinear PDE')
    # data
    parser.add_argument('--data-dir', type=str, default="./datasets", help='directory to dataset')
    parser.add_argument('--data', type=str, default='grf', choices=['grf', 'channelized', 'warped_grf'], help='data type')
    parser.add_argument('--kle', type=int, default=512, help='# kle terms')
    parser.add_argument('--imsize', type=int, default=64, help='image size')
    parser.add_argument('--idx', type=int, default=8, help='idx of input, please use 0 ~ 999')
    parser.add_argument('--alpha1', type=float, default=1.0, help='coefficient for the squared term')
    parser.add_argument('--alpha2', type=float, default=1.0, help='coefficient for the cubic term')
    # latent size: (nz, sz, sz)
    parser.add_argument('--nz', type=int, default=1, help='# feature maps of latent z')
    # parser.add_argument('--sz', type=int, default=16, help='feature map size of latent z')
    parser.add_argument('--blocks', type=list, default=[8, 6], help='# layers in each dense block of the decoder')
    parser.add_argument('--weight-bound', type=float, default=10, help='weight for boundary condition loss')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='# epochs to train')
    parser.add_argument('--test-freq', type=int, default=50, help='every # epoch to test')
    parser.add_argument('--ckpt-freq', type=int, default=250, help='every # epoch to save model')
    parser.add_argument('--cmap', type=str, default='jet', help='color map')
    parser.add_argument('--same-scale', action='store_true', help='true for setting noise to be same scale as output')
    parser.add_argument('--animate', action='store_true', help='true to plot animate figures')
    parser.add_argument('--cuda', type=int, default=1, help='cuda number')
    parser.add_argument('-v', '--verbose', action='store_true', help='True for versbose output')

    args = parser.parse_args()
    pprint(vars(args))
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    dataset = f'{args.data}_kle{args.kle}' if args.data == 'grf' else args.data
    hyparams = f'{dataset}_idx{args.idx}_dz{args.nz}_blocks{args.blocks}_'\
        f'lr{args.lr}_wb{args.weight_bound}_epochs{args.epochs}'
    
    if args.nonlinear:
        from utils.fenics import solve_nonlinear_poisson
        exp_name = 'conv_mixed_residual_nonlinear'
        from models.darcy import conv_constitutive_constraint_nonlinear as constitutive_constraint     
        hyparams = hyparams + f'_alpha1_{args.alpha1}_alpha2_{args.alpha2}'   
    else:
        exp_name = 'conv_mixed_residual'
        from models.darcy import conv_constitutive_constraint as constitutive_constraint

    run_dir = args.exp_dir + '/' + exp_name + '/' + hyparams
    mkdirs(run_dir)
    # load data
    assert args.idx < 1000
    if args.data == 'grf':
        assert args.kle in [512, 128, 1024, 2048]
        ntest = 1000 if args.kle == 512 else 1024
        hdf5_file = args.data_dir + f'/{args.imsize}x{args.imsize}/kle{args.kle}_lhs{ntest}_test.hdf5'
    elif args.data == 'warped_grf':
        hdf5_file = args.data_dir + f'/{args.imsize}x{args.imsize}/warped_gp_ng64_n1000.hdf5'
    elif args.data == 'channelized':
        hdf5_file = args.data_dir + f'/{args.imsize}x{args.imsize}/channel_ng64_n512_test.hdf5'
    else:
        raise ValueError('No dataset are found for the speficied parameters')
    print(f'dataset: {hdf5_file}')
    with h5py.File(hdf5_file, 'r') as f:
        input_data = f['input'][()]
        output_data = f['output'][()]
        print(f'input: {input_data.shape}')    
        print(f'output: {output_data.shape}') 
    # permeability, (1, 1, 64, 64)
    perm_arr = input_data[[args.idx]]
    # pressure, flux_hor, flux_ver, (3, 64, 64)
    if args.nonlinear:
        # solve nonlinear Darcy for perm_arr with FEniCS
        output_file = run_dir + '/output_fenics.npy'
        if os.path.isfile(output_file):
            output_arr = np.load(output_file)
            print('Loaded solved output field')
        else:
            print('Solve nonlinear poisson with FEniCS...')
            output_arr = solve_nonlinear_poisson(perm_arr[0, 0], args.alpha1, 
                args.alpha2, run_dir)
            np.save(output_file, output_arr)
    else:
        output_arr = output_data[args.idx]
    print('output shape: ', output_arr.shape)
    # model
    model = Decoder(args.nz, out_channels=3, blocks=args.blocks).to(device)
    print(f'model size: {model.model_size}')

    fixed_latent = torch.randn(1, args.nz, 16, 16).to(device) * 0.5
    perm_tensor = torch.FloatTensor(perm_arr).to(device)

    sobel_filter = SobelFilter(args.imsize, correct=True, device=device)
    optimizer = optim.LBFGS(model.parameters(), 
                            lr=args.lr, max_iter=20, history_size=50)

    logger = {}    
    logger['loss'] = []
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            output = model(fixed_latent)
            if args.nonlinear:
                energy = constitutive_constraint(perm_tensor, output, 
                    sobel_filter, args.alpha1, args.alpha2) \
                    + continuity_constraint(output, sobel_filter)
            else:
                energy = constitutive_constraint(perm_tensor, output, 
                    sobel_filter) + continuity_constraint(output, sobel_filter)
            loss_dirichlet, loss_neumann = boundary_condition(output)
            loss_boundary = loss_dirichlet + loss_neumann
            loss = energy + loss_boundary * args.weight_bound
            loss.backward()
            if args.verbose:
                print(f'epoch {epoch}: loss {loss.item():6f}, '\
                    f'energy {energy.item():.6f}, diri {loss_dirichlet.item():.6f}, '\
                    f'neum {loss_neumann.item():.6f}')
            return loss

        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        logger['loss'].append(loss_value)
        print(f'epoch {epoch}: loss {loss_value:.6f}')
        if epoch % args.ckpt_freq == 0:
            torch.save(model.state_dict(), run_dir + "/model_epoch{}.pth".format(epoch))

    def test(epoch):
        if epoch % args.epochs == 0 or epoch % args.test_freq == 0:
            output = model(fixed_latent)
            output = to_numpy(output)
            if args.animate:
                i_plot = epoch // args.test_freq
                plot_prediction_det_animate2(run_dir, output_arr, output[0], epoch, args.idx, i_plot,
                    plot_fn='imshow', cmap=args.cmap, same_scale=args.same_scale)
            else:
                plot_prediction_det(run_dir, output_arr, output[0], epoch, args.idx, 
                    plot_fn='imshow', cmap=args.cmap, same_scale=args.same_scale)
            np.save(run_dir + f'/epoch{epoch}.npy', output[0])
            
    print('start training...')
    dryrun = False
    tic = time.time()
    for epoch in range(1, args.epochs + 1):
        if not dryrun:
            train(epoch)
        test(epoch)
    print(f'Finished optimization for {args.epochs} epochs using {(time.time()-tic)/60:.3f} minutes')
    save_stats(run_dir, logger, 'loss')
    # save input
    plt.imshow(perm_arr[0, 0])
    plt.colorbar()
    plt.savefig(run_dir + '/input.png')
    plt.close()

if __name__ == '__main__':
    main()
