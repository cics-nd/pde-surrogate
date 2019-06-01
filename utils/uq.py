import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.io
from scipy.stats import norm as scipy_norm
import seaborn as sns
from utils.misc import mkdir, to_numpy
from utils.plot import plot_prediction_bayes2, plot_MC2, save_samples
from utils.lhs import lhs
plt.switch_backend('agg')


class UQ_CondGlow(object):
    """Class for uncertainty quantification tasks, include:
    
    - prediction at one input realization
    - uncertainty propagation
    - distribution estimate at certain location
    - reliability diagram (assess uncertainty quality)
    Args:
        model: Pre-trained probabilistic surrogate
        args: training arguments
        mc_loader (utils.data.DataLoader): Dataloader for Monte Carlo data
    """
    def __init__(self, model, args, mc_loader, test_loader, y_test_variation, 
        n_samples=20, temperature=1.0):
        self.model = model
        self.mc_loader = mc_loader
        self.test_loader = test_loader
        self.y_test_variation = y_test_variation
        self.ntrain = args.ntrain
        self.plot_fn = args.plot_fn
        self.epochs = args.epochs
        self.device = args.device
        self.post_dir = args.post_dir
        self.imsize = args.imsize
        self.n_samples = n_samples
        self.temperature = temperature

        print(f'mc loader size: {len(self.mc_loader.dataset)}')
        print(f'test loader size: {len(self.test_loader.dataset)}')


    def plot_prediction_at_x(self, n_pred, plot_samples=False):
        r"""Plot `n_pred` predictions for randomly selected input from test dataset.
        - target
        - predictive mean
        - standard deviation of predictive output distribution
        - error of the above two

        Args:
            n_pred: number of candidate predictions
            plot_samples (bool): plot 15 output samples from p(y|x) for given x
        """
        save_dir = self.post_dir + '/predict_at_x'
        mkdir(save_dir)
        print('Plotting predictions at x from test dataset..................')
        np.random.seed(1)
        idx = np.random.permutation(len(self.test_loader.dataset))[:n_pred]
        for i in idx:
            print('input index: {}'.format(i))
            input, target = self.test_loader.dataset[i]
            pred_mean, pred_var = self.model.predict(input.unsqueeze(0).to(self.device),
                n_samples=self.n_samples, temperature=self.temperature)
            
            plot_prediction_bayes2(save_dir, target, pred_mean.squeeze(0), 
                pred_var.squeeze(0), self.epochs, i, plot_fn=self.plot_fn)
            if plot_samples:
                samples_pred = self.model.sample(input.unsqueeze(0).to(self.device), 
                    n_samples=15)[:, 0]
                samples = torch.cat((target.unsqueeze(0), samples_pred.detach().cpu()), 0)
                save_samples(save_dir, samples, self.epochs, i, 
                    'samples', nrow=4, heatmap=True, cmap='jet')


    def propagate_uncertainty(self, manual_scale=False, var_samples=10):
        print("Propagate Uncertainty using pre-trained surrogate ...........")
        # compute MC sample mean and variance in mini-batch
        sample_mean_x = torch.zeros_like(self.mc_loader.dataset[0][0])
        sample_var_x = torch.zeros_like(sample_mean_x)
        sample_mean_y = torch.zeros_like(self.mc_loader.dataset[0][1])
        sample_var_y = torch.zeros_like(sample_mean_y)

        for _, (x_test_mc, y_test_mc) in enumerate(self.mc_loader):
            x_test_mc, y_test_mc = x_test_mc, y_test_mc
            sample_mean_x += x_test_mc.mean(0)
            sample_mean_y += y_test_mc.mean(0)
        sample_mean_x /= len(self.mc_loader)
        sample_mean_y /= len(self.mc_loader)

        for _, (x_test_mc, y_test_mc) in enumerate(self.mc_loader):
            x_test_mc, y_test_mc = x_test_mc, y_test_mc
            sample_var_x += ((x_test_mc - sample_mean_x) ** 2).mean(0)
            sample_var_y += ((y_test_mc - sample_mean_y) ** 2).mean(0)
        sample_var_x /= len(self.mc_loader)
        sample_var_y /= len(self.mc_loader)

        # plot input MC
        stats_x = torch.stack((sample_mean_x, sample_var_x)).cpu().numpy()
        fig, _ = plt.subplots(1, 2)
        for i, ax in enumerate(fig.axes):
            # ax.set_title(titles[i])
            ax.set_aspect('equal')
            ax.set_axis_off()
            # im = ax.imshow(stats_x[i].squeeze(0),
            #                interpolation='bilinear', cmap=self.args.cmap)
            im = ax.contourf(stats_x[i].squeeze(0), 50, cmap='jet')
            for c in im.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
        out_stats_dir = self.post_dir + '/out_stats'
        mkdir(out_stats_dir)
        plt.savefig(out_stats_dir + '/input_MC.pdf', di=300, bbox_inches='tight')
        plt.close(fig)
        print("Done plotting input MC, num of training: {}".format(self.ntrain))

        # MC surrogate predictions
        y_pred_EE, y_pred_VE, y_pred_EV, y_pred_VV = self.model.propagate(
                self.mc_loader, n_samples=self.n_samples, 
                temperature=self.temperature, var_samples=var_samples)
        print('Done MC predictions')

        # plot the 4 output stats
        # plot the predictive mean
        plot_MC2(out_stats_dir, sample_mean_y, y_pred_EE, y_pred_VE, True, 
            self.ntrain, manual_scale=manual_scale)
        # plot the predictive var
        plot_MC2(out_stats_dir, sample_var_y, y_pred_EV, y_pred_VV, False, 
            self.ntrain)

        # save for MATLAB plotting
        scipy.io.savemat(out_stats_dir + '/out_stats.mat',
                         {'sample_mean': sample_mean_y.cpu().numpy(),
                          'sample_var': sample_var_y.cpu().numpy(),
                          'y_pred_EE': y_pred_EE.cpu().numpy(),
                          'y_pred_VE': y_pred_VE.cpu().numpy(),
                          'y_pred_EV': y_pred_EV.cpu().numpy(),
                          'y_pred_VV': y_pred_VV.cpu().numpy()})
        print('saved output stats to .mat file')


    def plot_dist(self, num_loc):
        """Plot distribution estimate in `num_loc` locations in the domain, 
        which are chosen by Latin Hypercube Sampling.
        Args:
            num_loc (int): number of locations where distribution is estimated
        """
        print('Plotting distribution estimate.................................')

        assert num_loc > 0, 'num_loc must be greater than zero'
        locations = lhs(2, num_loc, criterion='c')
        print('Locations selected by LHS: \n{}'.format(locations))
        # location (ndarray): [0, 1] x [0, 1]: N x 2
        idx = (locations * self.imsize).astype(int)

        print('Propagating...')
        pred, target = [], []
        for _, (x_mc, t_mc) in enumerate(self.mc_loader):
            x_mc = x_mc.to(self.device)
            # S x B x C x H x W
            y_mc = self.model.sample(x_mc, n_samples=self.n_samples, 
                temperature=self.temperature)
            # S x B x C x n_points
            pred.append(y_mc[:, :, :, idx[:, 0], idx[:, 1]])
            # B x C x n_points
            target.append(t_mc[:, :, idx[:, 0], idx[:, 1]])
        # S x M x C x n_points --> M x C x n_points
        pred = torch.cat(pred, dim=1).mean(0).cpu().numpy()

        print('pred size: {}'.format(pred.shape))
        # M x C x n_points
        target = torch.cat(target, dim=0).cpu().numpy()
        print('target shape: {}'.format(target.shape))
        dist_dir = self.post_dir + '/dist_estimate'
        mkdir(dist_dir)
        for loc in range(locations.shape[0]):
            print(loc)
            fig, _ = plt.subplots(1, 3, figsize=(12, 4))
            for c, ax in enumerate(fig.axes):
                sns.kdeplot(target[:, c, loc], color='b', ls='--', label='Monte Carlo', ax=ax)
                sns.kdeplot(pred[:, c, loc], color='r', label='Surrogate', ax=ax)
                ax.legend()
            plt.savefig(dist_dir + '/loc_({:.5f}, {:.5f}).pdf'
                        .format(locations[loc][0], locations[loc][1]), dpi=300)
            plt.close(fig)


    def plot_reliability_diagram(self, label='Conditional Glow', save_time=True):
        print("Plotting reliability diagram..................................")
        # percentage: p
        # p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        p_list = np.linspace(0.01, 0.99, 10)
        freq = []
        n_channels = self.mc_loader.dataset[0][1].shape[0]

        for p in p_list:
            count = 0
            numels = 0
            for batch_idx, (input, target) in enumerate(self.mc_loader):
                # only evaluate 2000 of the MC data to save time
                if save_time and batch_idx > 4:
                    continue
                pred_mean, pred_var = self.model.predict(input.to(self.device), 
                    n_samples=self.n_samples, temperature=self.temperature)

                interval = scipy_norm.interval(p, loc=pred_mean.cpu().numpy(),
                                            scale=pred_var.sqrt().cpu().numpy())

                count += ((target.numpy() >= interval[0])
                          & (target.numpy() <= interval[1])).sum(axis=(0, 2, 3))
                numels += target.numel() / n_channels
                print('p: {}, {} / {} = {}'.format(p, count, numels, 
                    np.true_divide(count, numels)))
            freq.append(np.true_divide(count, numels))
        reliability_dir = self.post_dir + '/uncertainty_quality'
        mkdir(reliability_dir)

        freq = np.stack(freq, 0)
        for i in range(freq.shape[-1]):
            plt.figure()
            plt.plot(p_list, freq[:, i], 'r', label=label)
            plt.xlabel('Probability')
            plt.ylabel('Frequency')
            x = np.linspace(0, 1, 100)
            plt.plot(x, x, 'k--', label='Ideal')
            plt.legend(loc='upper left')
            plt.savefig(reliability_dir + f"/reliability_diagram_{i}.pdf", dpi=300)
            plt.close()

        reliability = np.zeros((p_list.shape[0], 1+n_channels))
        reliability[:, 0] = p_list
        reliability[:, 1:] = freq
        np.savetxt(reliability_dir + "/reliability_diagram.txt", reliability)
        plt.close()


    def test_metric(self, handle_nan=True):
        relative_l2, err2 = [], []
        num_nan_inf = 0
        for batch_idx, (input, target) in enumerate(self.test_loader):
            input, target = input.to(self.device), target.to(self.device)
            pred_mean, pred_var = self.model.predict(input, n_samples=self.n_samples, 
                temperature=self.temperature)
            # handling nan, inf
            if handle_nan:
                exception = torch.isnan(pred_mean) + torch.isinf(pred_mean)
                exception = exception.sum((1, 2, 3)).gt(0)
                normal = (1 - exception)
                # print(normal)
                normal_idx = torch.arange(len(normal)).to(self.device).masked_select(normal).to(torch.long)
                # print(normal_idx)
                pred_mean, target = pred_mean.index_select(0, normal_idx), target.index_select(0, normal_idx)
                num_nan_inf += exception.sum()
                # print(pred_mean.shape)

            err2_sum = torch.sum((pred_mean - target) ** 2, [-1, -2])
            relative_l2.append(torch.sqrt(err2_sum / (target ** 2).sum([-1, -2])))
            err2.append(err2_sum)

        relative_l2 = to_numpy(torch.cat(relative_l2, 0).mean(0))
        r2_score = 1 - to_numpy(torch.cat(err2, 0).sum(0)) / self.y_test_variation
        print(relative_l2)
        print(r2_score)
        np.savetxt(self.post_dir + '/nrmse_test.txt', relative_l2)
        np.savetxt(self.post_dir + '/r2_test.txt', r2_score)
        if handle_nan:
            abnormal_rate = num_nan_inf / len(self.test_loader.dataset)
            print(f'num_nan_inf: {num_nan_inf}')
            print(f'abnormal rate: {abnormal_rate:.6f}')
            np.savetxt(self.post_dir + '/log_stats.txt', 
                [num_nan_inf, len(self.test_loader.dataset), abnormal_rate])
        