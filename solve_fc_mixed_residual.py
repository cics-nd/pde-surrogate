r"""Solving Darcy Flow using fully-connected neural nets with mixed residual loss
Flow through Porous Media, 2D

div (K(x) grad u(x)) = 0, x = (x1, x2) \in (0, 1) x (0, 1)
Boundary:   
    u = 1, s1 = 0; u = 0, s1 = 1
    u_s2 = 0, s2 in {0, 1}

K -- permeability
u -- pressure

Optimizer: L-BFGS
Only considered the linear PDE case.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.cppn import CPPN
from utils.sampling import SampleSpatial2d
from utils.image_gradient import SobelFilter
from models.darcy import mixed_residual_fc, neumann_boundary_mixed, grad
from utils.plot import plot_row, save_stats, plot_prediction_det_animate2, plot_prediction_det
from utils.misc import mkdirs

import numpy as np
import argparse
import h5py
import sys
import time
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
print('ok')

def main():
    parser = argparse.ArgumentParser(description='CNN to solve PDE')
    parser.add_argument('--exp-dir', type=str, default='./experiments/solver', help='color map')
    # data
    parser.add_argument('--data-dir', type=str, default="./datasets", help='directory to dataset')
    parser.add_argument('--data', type=str, default='grf', choices=['grf', 'channelized', 'warped_grf'], help='data type')
    parser.add_argument('--kle', type=int, default=512, help='# kle terms')
    parser.add_argument('--imsize', type=int, default=64, help='image size')
    parser.add_argument('--idx', type=int, default=8, help='idx of input, please use 0 ~ 999')
    parser.add_argument('--alpha1', type=float, default=1.0, help='coefficient for the squared term')
    parser.add_argument('--alpha2', type=float, default=1.0, help='coefficient for the cubic term')
    parser.add_argument('--dim-hidden', type=int, default=512, help='# nodes in each hidden layer')
    parser.add_argument('--layers-hidden', type=int, default=8, help='# hidden layers')
    parser.add_argument('--off-grid', action='store_true', help='set True to use colloc ')
    parser.add_argument('--n-colloc', type=int, default=4096, help='# collocation points')
    parser.add_argument('--weight-bound', type=float, default=10, help='weight for boundary condition loss')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2000, help='# epochs to train')
    parser.add_argument('--test-freq', type=int, default=50, help='every # epoch to test')
    parser.add_argument('--ckpt-freq', type=int, default=250, help='every # epoch to save model')
    parser.add_argument('--cmap', type=str, default='jet', help='color map')
    parser.add_argument('--same-scale', action='store_true', help='true for setting noise to be same scale as output')
    parser.add_argument('--animate', action='store_true', help='true to plot animate figures')
    parser.add_argument('--cuda', type=int, default=2, help='cuda number')
    parser.add_argument('-v', '--verbose', action='store_true', help='True for versbose output')


    args = parser.parse_args()
    pprint(vars(args))
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
 
    exp_name = 'fc_mixed_residual'
    dataset = f'{args.data}_kle{args.kle}' if args.data == 'grf' else args.data
    hyparams = f'{dataset}_idx{args.idx}_dhid{args.dim_hidden}_lhid{args.layers_hidden}_alpha1_{args.alpha1}_alpha2_{args.alpha2}_'\
    f'lr{args.lr}_wb{args.weight_bound}_epochs{args.epochs}_ongrid_{not args.off_grid}_ncolloc{args.n_colloc}'

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
        assert args.idx < 512
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
    output_arr = output_data[args.idx]

    def to_tensor_gpu(*numpy_seq):
        # x: numpy array --> tensor on GPU
        return (torch.FloatTensor(x).to(device) for x in numpy_seq)

    # define networks
    net_u = CPPN(dim_in=2, dim_out=3, dim_hidden=args.dim_hidden, 
        layers_hidden=args.layers_hidden).to(device)
    print(net_u)
    print(net_u._model_size())
    optimizer = optim.LBFGS(net_u.parameters(), 
                            lr=args.lr, max_iter=20, history_size=50)
            
    logger = {}    
    logger['loss'] = []
    ngrids = [args.imsize, args.imsize]
    sampler = SampleSpatial2d(int(ngrids[0]), int(ngrids[1]))
    colloc_on_grid = not args.off_grid
    # for batch optimization
    x_colloc = sampler.colloc(colloc_on_grid, n_samples=args.n_colloc).to(device)
    x_dirichlet = torch.cat((sampler.left(on_grid=False, n_samples=256), 
        sampler.right(on_grid=False, n_samples=256)), 0).to(device)
    y_dirichlet = torch.cat((torch.ones(256, 1), torch.zeros(256, 1)), 0).to(device)
    x_neumann = torch.cat((sampler.top(colloc_on_grid), 
        sampler.bottom(colloc_on_grid)), 0).to(device)
    print(sampler.coordinates_no_boundary.shape)

    K_true_tensor, = to_tensor_gpu(perm_arr.reshape(-1, 1))
    if args.verbose:
        print('x_colloc: {}'.format(x_colloc.shape))
        print('x_dirc: {}'.format(x_dirichlet.shape))
        print('y_dirc: {}'.format(y_dirichlet.shape))
        print('x_neumann: {}'.format(x_neumann.shape))

    def train(epoch):
        net_u.train()
        def closure():
            optimizer.zero_grad()
            loss_colloc = mixed_residual_fc(net_u, x_colloc, K_true_tensor, 
                args.verbose, rand_colloc=args.off_grid)
            # loss_colloc = 0
            loss_dirichlet = F.mse_loss(net_u(x_dirichlet)[:, [0]], y_dirichlet)
            loss_neumann = neumann_boundary_mixed(net_u, x_neumann)
            loss = loss_colloc + args.weight_bound * (loss_dirichlet + loss_neumann)
            loss.backward()
            if args.verbose:
                print(f'epoch {epoch}: colloc {loss_colloc:.6f}, '
                    f'diri {loss_dirichlet:.6f}, neum {loss_neumann:.6f}')
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        logger['loss'].append(loss_value)
        print('epoch {}: loss {:.10f}'.format(epoch, loss_value))

        if epoch % args.ckpt_freq == 0:
            torch.save(net_u.state_dict(), run_dir + "/model_epoch{}.pth".format(epoch))
                
    def test(epoch):
        if epoch % args.epochs == 0 or epoch % args.test_freq == 0:
            # plot the solution
            xx, yy = np.meshgrid(np.arange(ngrids[0]), np.arange(ngrids[1]))
            x_test = xx.flatten()[:, None] / ngrids[1]
            y_test = yy.flatten()[:, None] / ngrids[0]
            x_test, y_test = to_tensor_gpu(x_test, y_test)
            net_u.eval()
            x_test.requires_grad = True
            y_test.requires_grad = True
            xy_test = torch.cat((y_test, x_test), 1)
            y_pred = net_u(xy_test)
            target = output_arr
            # three output of net_u from 0-3 channel: u, flux_y, flux_x
            u_pred = y_pred[:, 0].detach().cpu().numpy().reshape(*ngrids)
            u_y = y_pred[:, 1].detach().cpu().numpy().reshape(*ngrids)
            u_x = y_pred[:, 2].detach().cpu().numpy().reshape(*ngrids)
            prediction = np.stack((u_pred, u_x, u_y))
            # prediction = y_pred.view(*ngrids, -1).transpose(0, 1).permute(2, 1, 0).detach().cpu().numpy()
            if args.animate:
                i_plot = epoch // args.test_freq
                plot_prediction_det_animate2(run_dir, target, prediction, epoch, args.idx, i_plot,
                    plot_fn='imshow', cmap=args.cmap, same_scale=args.same_scale)
            else:
                plot_prediction_det(run_dir, target, prediction, epoch, args.idx, 
                    plot_fn='imshow', cmap=args.cmap, same_scale=args.same_scale)
            np.save(run_dir + f'/epoch{epoch}.npy', prediction)
           
    print('start training...')
    dryrun = False
    tic = time.time()
    for epoch in range(1, args.epochs + 1):
        if not dryrun:
            train(epoch)
        test(epoch)
    print(f'Finished training {args.epochs} epochs in {(time.time()-tic)/60:.3f} minutes')
    save_stats(run_dir, logger, 'loss')

    # save input
    plt.close()
    plt.imshow(np.log(perm_arr[0, 0]))
    plt.colorbar()
    plt.savefig(run_dir + '/input_logK.png')
    plt.close()

    # super-resultion one
    ngrids = (640, 640)
    xx, yy = np.meshgrid(np.arange(ngrids[0]), np.arange(ngrids[1]))
    x_test = torch.FloatTensor(np.stack((yy.flatten() / (ngrids[1]-1), 
        xx.flatten() / (ngrids[0]-1)), 1)).to(device)
    net_u.eval()
    u_pred = net_u(x_test)
    u_pred = u_pred[:, 0].reshape(*ngrids).detach().cpu().numpy() 
    plt.contourf(u_pred, 65)
    plt.colorbar()
    plt.savefig(run_dir + '/solution_HR.png')
    plt.close()

if __name__ == '__main__':
    main()