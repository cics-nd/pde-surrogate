"""Training Multiscale conditional Glow with reverse KL divervence loss.
Training does not require output data.

"""
import torch
from torch import autograd
import numpy as np
import torch.optim as optim
from models.glow_msc import MultiScaleCondGlow
from models.darcy import conv_constitutive_constraint as constitutive_constraint
from models.darcy import conv_continuity_constraint as continuity_constraint
from models.darcy import conv_boundary_condition as boundary_condition
from utils.image_gradient import SobelFilter
from utils.load import load_data
from utils.misc import mkdirs, to_numpy
from utils.plot import (save_stats, save_samples, plot_prediction_det, 
    plot_prediction_bayes2)
from utils.practices import OneCycleScheduler, adjust_learning_rate, find_lr
import argparse
import time
import random
from pprint import pprint
import json
import math
import os


class Parser(argparse.ArgumentParser):
    def __init__(self): 
        super(Parser, self).__init__(description='Training multiscale conditional Glows with reverse KLD loss')
        self.add_argument('--exp-name', type=str, default='cglow/reverse_kld', help='experiment name')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')        
        # cglow
        self.add_argument('--enc-blocks', type=list, default=[3, 4, 4], help='list of number of layers in each dense block of the encoder')
        self.add_argument('--flow-blocks', type=list, default=[6, 6, 6], help='list of number of layers in each revblock')
        # self.add_argument('--enc-blocks', type=list, default=[3, 3, 3, 3], help='list of number of layers in each dense block')
        # self.add_argument('--flow-blocks', type=list, default=[4, 4, 4, 4], help='list of number of layers in each revblock')
        self.add_argument('--no-LU-decompose', action='store_true', default=False, help='Use LU decomposition to parameters invertible 1x1 conv')
        # data 
        self.add_argument('--data-dir', type=str, default="./datasets", help='directory to dataset')
        self.add_argument('--kle', type=int, default=100, help='num of KLE terms')
        self.add_argument('--ntrain', type=int, default=4096, help="number of training data")
        self.add_argument('--ntest', type=int, default=512, help="number of test data")
        self.add_argument('--x-channels', type=int, default=1)
        self.add_argument('--y-channels', type=int, default=3)
        self.add_argument('--imsize', type=int, default=32)
        # training
        self.add_argument('--data-init', action='store_true', default=False, help='use data initialization for ActNorm')
        self.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=1.5e-3, help='learning rate')
        self.add_argument('--lr-div', type=float, default=2., help='lr div factor to get the initial lr')
        self.add_argument('--lr-pct', type=float, default=0.3, help='percentage of epochs to reach the (max) lr')
        self.add_argument('--beta', type=float, default=150, help='precision parameter in Boltzmann distribution')
        self.add_argument('--weight-decay', type=float, default=0., help="weight decay")
        self.add_argument('--weight-bound', type=float, default=50, help="weight for boundary loss")
        self.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 16)')
        self.add_argument('--test-batch-size', type=int, default=64, help='input batch size for testing (default: 100)')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
        self.add_argument('--cuda', type=int, default=1, help='No. of GPU card to use, choices=[0, 1, 2, 3] on CRC')
        # logging
        self.add_argument('--debug', action='store_true', default=False, help='debug or verbose')
        self.add_argument('--resume', action='store_true', default=False, help='resume training from the lastest checkpoint')
        self.add_argument('--ckpt-epoch', type=int, default=None, help='resume training from the checkpoint at this epoch')
        self.add_argument('--ckpt-freq', type=int, default=25, help='how many epochs to wait before saving model')
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-freq', type=int, default=25, help='how many epochs to wait before plotting test output')
        self.add_argument('--plot-fn', type=str, default='imshow', choices=['contourf', 'imshow'], help='plotting method')

    def parse(self):
        args = self.parse_args()
        args.LU_decompose = not args.no_LU_decompose
        assert len(args.enc_blocks) == len(args.flow_blocks)
        hparams = f'kle{args.kle}_ntrain{args.ntrain}_'\
            f'ENC_blocks{args.enc_blocks}_FLOW_blocks{args.flow_blocks}_'\
            f'wb{args.weight_bound}_beta{args.beta}_'\
            f'batch{args.batch_size}_lr{args.lr}_epochs{args.epochs}'
        if args.debug:
            hparams = 'debug/' + hparams
        if args.data_init:
            hparams = hparams + '_data_init'
        args.run_dir = args.exp_dir + '/' + args.exp_name + '/' + hparams
        args.ckpt_dir = args.run_dir + '/checkpoints'
        args.train_dir = args.run_dir + '/training'
        args.pred_dir = args.train_dir + '/predictions'
        mkdirs(args.run_dir, args.ckpt_dir, args.train_dir, args.pred_dir)

        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        args_file = args.run_dir + "/args.txt"
        if os.path.isfile(args_file):
            # args.resume, args.ckpt directly overwrite
            if args.ckpt_epoch is None and args.resume: 
                with open(args_file, 'r') as args_f:
                    args_old = argparse.Namespace(**json.load(args_f))
                args.ckpt_epoch = args_old.ckpt_epoch
        else:
            with open(args_file, 'w') as args_f:
                json.dump(vars(args), args_f, indent=4)
        print('Arguments:')
        pprint(vars(args))
        return args


if __name__ == '__main__':

    args = Parser().parse()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # load input
    train_hdf5_file = args.data_dir + \
        f'/{args.imsize}x{args.imsize}/kle{args.kle}_lhs10000_train.hdf5'
    train_loader, _ = load_data(train_hdf5_file, args.ntrain, args.batch_size, 
        only_input=True, return_stats=False)
    # load val set
    test_hdf5_file = args.data_dir + \
        f'/{args.imsize}x{args.imsize}/kle{args.kle}_lhs1000_val.hdf5'
    test_loader, test_stats = load_data(test_hdf5_file, args.ntest, 
        args.test_batch_size, only_input=False, return_stats=True)
    y_test_variation = test_stats['y_variation']
    print(f'Test output variation per channel: {y_test_variation}')

    n_out_pixels = test_loader.dataset[0][1].numel()
    print(f'# out pixels per output: {n_out_pixels}')

    model = MultiScaleCondGlow(img_size=args.imsize, 
                            x_channels=args.x_channels, 
                            y_channels=args.y_channels, 
                            enc_blocks=args.enc_blocks, 
                            flow_blocks=args.flow_blocks, 
                            LUdecompose=args.LU_decompose,
                            squeeze_factor=2,
                            data_init=args.data_init).to(device)
    if args.debug:
        print(model)
    print(model.model_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)
    scheduler = OneCycleScheduler(lr_max=args.lr, div_factor=args.lr_div, 
                        pct_start=args.lr_pct)
    sobel_filter = SobelFilter(args.imsize, correct=True, device=device)

    logger = {}
    logger['loss_train'] = []
    logger['loss_test'] = []
    logger['nrmse_test'] = []
    logger['r2_test'] = []
    logger['entropy_train'] = []
    logger['entropy_test'] = []

    if args.ckpt_epoch is not None:
        ckpt_file = args.ckpt_dir + f"/model_epoch{args.ckpt_epoch}.pth"
        checkpoint = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])       
        if args.data_init:
            model.init_actnorm()
        logger = checkpoint['logger']
        print(f'Loaded checkpoint at epoch {args.ckpt_epoch}')

    def test(epoch):
        model.eval()
        loss_test = 0.
        # mse = 0.
        relative_l2, err2 = [], []
        for batch_idx, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)
            # every 10 epochs evaluate the mean accurately
            if epoch % 10 == 0:
                output_samples = model.sample(input, n_samples=20, temperature=1.0)
                output = output_samples.mean(0)
            else:
                # evaluate with one output sample
                output, _ = model.generate(input)

            residual_norm = constitutive_constraint(input, output, sobel_filter) \
                + continuity_constraint(output, sobel_filter)
            loss_dirichlet, loss_neumann = boundary_condition(output)
            loss_boundary = loss_dirichlet + loss_neumann
            loss_pde = residual_norm + loss_boundary * args.weight_bound
            # evaluate predictive entropy: E_p(y|x) [log p(y|x)]
            neg_entropy = log_likeihood.mean() / math.log(2.) / n_out_pixels
            loss = loss_pde * args.beta + neg_entropy

            loss_test += loss.item()
            err2_sum = torch.sum((output - target) ** 2, [-1, -2])
            # print(err2_sum)
            relative_l2.append(torch.sqrt(err2_sum / (target ** 2).sum([-1, -2])))
            err2.append(err2_sum)

            # plot predictions
            if (epoch % args.plot_freq == 0 or epoch % args.epochs == 0) and batch_idx == 0:
                n_samples = 6 if epoch == args.epochs else 2
                idx = np.random.permutation(input.size(0))[:n_samples]
                samples_target = target.data.cpu()[idx].numpy()

                for i in range(n_samples):
                    print('epoch {}: plotting prediction {}'.format(epoch, i))
                    pred_mean, pred_var = model.predict(input[[idx[i]]])
                    plot_prediction_bayes2(args.pred_dir, samples_target[i], 
                        pred_mean[0], pred_var[0], epoch, idx[i], 
                        plot_fn='imshow', cmap='jet', same_scale=False)
                    # plot samples p(y|x)
                    print(idx[i])
                    print(input[[idx[i]]].shape)
                    samples_pred = model.sample(input[[idx[i]]], n_samples=15)[:, 0]
                    samples = torch.cat((target[[idx[i]]], samples_pred), 0)
                    # print(samples.shape)
                    save_samples(args.pred_dir, samples, epoch, idx[i], 
                        'samples', nrow=4, heatmap=True, cmap='jet')                    

        loss_test /= (batch_idx + 1)
        relative_l2 = to_numpy(torch.cat(relative_l2, 0).mean(0))
        r2_score = 1 - to_numpy(torch.cat(err2, 0).sum(0)) / y_test_variation

        print(f"Epoch {epoch}: test r2-score:  {r2_score}")
        print(f"Epoch {epoch}: test relative l2:  {relative_l2}")
        print(f'Epoch {epoch}: test loss: {loss_test:.6f}, residual: {residual_norm.item():.6f}, '\
                f'boundary {loss_boundary.item():.6f}, neg entropy {neg_entropy.item():.6f}')

        if epoch % args.log_freq == 0:
            logger['loss_test'].append(loss_test)
            logger['r2_test'].append(r2_score)
            logger['nrmse_test'].append(relative_l2)
            logger['entropy_test'].append(-neg_entropy.item())

    print('Start training........................................................')
    tic = time.time()
    start_epoch = checkpoint['epoch']+1 if args.ckpt_epoch is not None else 1
    initialized = False if start_epoch == 1 else True
    total_steps = (args.epochs - start_epoch) * len(train_loader)
    print(f'total steps: {total_steps}')
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_train, mse = 0., 0.
        if args.data_init and (not initialized):
            # data initialization for actnorm only when `--data-init` is set
            # By default it is not used
            for batch_idx, (input, target) in enumerate(test_loader):
                input, target = input.to(device), target.to(device)
                model.zero_grad()
                latent, logp, eps = model(target, input)
                break
            initialized = True
            print('Finished data initialization of Actnorm')

        for batch_idx, (input,) in enumerate(train_loader):
            input = input.to(device)
            model.zero_grad()
            # sample output for each input
            with autograd.detect_anomaly():
                output, log_likeihood = model.generate(input)
                # evaluate energy functional/residual norm: E_x E_p(y|x) [\beta V(y; x)]
                residual_norm = constitutive_constraint(input, output, sobel_filter) \
                    + continuity_constraint(output, sobel_filter)
                loss_dirichlet, loss_neumann = boundary_condition(output)
                loss_boundary = loss_dirichlet + loss_neumann
                loss_pde = residual_norm + loss_boundary * args.weight_bound
                # evaluate predictive entropy: E_p(y|x) [log p(y|x)], bits per pixel
                neg_entropy = log_likeihood.mean() / math.log(2.) / n_out_pixels
                loss = loss_pde * args.beta + neg_entropy
                loss.backward()

            step = (epoch - 1) * len(train_loader) + batch_idx
            pct = step / total_steps
            lr = scheduler.step(pct)
            adjust_learning_rate(optimizer, lr)

            optimizer.step()
            loss_train += loss.item()

        loss_train /= (batch_idx + 1) 
        print(f'Epoch {epoch}: training loss: {loss_train:.6f}, residual: {residual_norm.item():.6f}, '\
            f'boundary {loss_boundary.item():.6f}, neg entropy {neg_entropy.item():.6f}')
        if epoch % args.log_freq == 0:
            logger['loss_train'].append(loss_train)
            logger['entropy_train'].append(-neg_entropy.item())
        if epoch % args.ckpt_freq == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'logger': logger
                        }, args.ckpt_dir + f"/model_epoch{epoch}.pth")
            args.ckpt_epoch = epoch
            with open(args.run_dir + "/args.txt", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        with torch.no_grad():
            test(epoch)

    tic2 = time.time()
    print(f'Finished training {args.epochs} epochs with {args.ntrain} data '\
        f'using {(tic2 - tic) / 60:.2f} mins')
    metrics = ['loss_train', 'loss_test', 'nrmse_test', 'r2_test', 
        'entropy_test', 'entropy_train']
    save_stats(args.train_dir, logger, *metrics)
    args.training_time = tic2 - tic
    args.n_params, args.n_layers = model.model_size
    with open(args.run_dir + "/args.txt", 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)
