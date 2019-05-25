import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
import numpy as np
from .misc import to_numpy
plt.switch_backend('agg')

pub = False
if pub:
    ext = 'pdf'
    dpi = 300
else:
    ext = 'png'
    dpi = None


def plot_prediction_det(save_dir, target, prediction, epoch, index, 
                        plot_fn='contourf', cmap='jet', same_scale=False, row_labels=None, col_labels=None):
    """Plot prediction for one input (`index`-th at epoch `epoch`)
    Args:
        save_dir: directory to save predictions
        target (np.ndarray): (3, 65, 65)
        prediction (np.ndarray): (3, 65, 65)
        epoch (int): which epoch
        index (int): i-th prediction
        plot_fn (str): choices=['contourf', 'imshow']
    """
    target, prediction = to_numpy(target), to_numpy(prediction)
    
    if row_labels is not None:
        rows = row_labels
    else:
        rows = ['Simulation', 'Prediction', r'Simulation $-$ Prediction']
    if col_labels is not None:
        cols = col_labels
    else:
        cols = ['Pressure', 'Horizontal Flux', 'Vertical Flux']

    # 3 x 65 x 65
    n_fields = target.shape[0]
    samples = np.concatenate((target, prediction, target - prediction), axis=0)
    # print(samples.shape)
    interp = None
    vmin, vmax = [], []
    for i in range(n_fields):
        vmin.append(np.amin(samples[[i, i+n_fields]]))
        vmax.append(np.amax(samples[[i, i+n_fields]]))

    fig, axes = plt.subplots(3, n_fields, figsize=(3.75 * n_fields, 9))
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        if j < 2 * n_fields:
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap=cmap,
                                  vmin=vmin[j % n_fields], vmax=vmax[j % n_fields])
            elif plot_fn =='imshow':
                cax = ax.imshow(samples[j], cmap=cmap, origin='upper',
                                interpolation=interp,
                                vmin=vmin[j % n_fields], vmax=vmax[j % n_fields])   
        else:
            if same_scale:
                vmin_error, vmax_error = vmin[j % n_fields], vmax[j % n_fields]
            else:
                vmin_error, vmax_error = None, None
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap=cmap)
            elif plot_fn =='imshow':
                cax = ax.imshow(samples[j], cmap=cmap, origin='upper',
                                interpolation=interp, vmin=vmin_error, vmax=vmax_error)
        if plot_fn == 'contourf':
            for c in cax.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.ax.yaxis.set_offset_position('left')
        # cbar.ax.tick_params(labelsize=5)
        cbar.update_ticks()
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, size='large')

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    # plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    # plt.subplots_adjust(top=0.93)
    plt.savefig(save_dir + '/pred_epoch{}_{}.{}'.format(epoch, index, ext),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_prediction_det_animate2(save_dir, target, prediction, epoch, index, i_plot,
                        plot_fn='imshow', cmap='jet', same_scale=False, 
                        vmax=None, vmin=None, vmax_err=None, vmin_err=None):
    """Plot prediction for one input (`index`-th at epoch `epoch`)
    Args:
        save_dir: directory to save predictions
        target (np.ndarray): (3, 65, 65)
        prediction (np.ndarray): (3, 65, 65)
        epoch (int): which epoch
        index (int): i-th prediction
        plot_fn (str): choices=['contourf', 'imshow']
    """
    target, prediction = to_numpy(target), to_numpy(prediction)
    
    rows = ['Simulation', 'Prediction', 'Abs Error']
    cols = ['Pressure', 'Horizontal Flux', 'Vertical Flux']

    # 3 x 65 x 65
    n_fields = target.shape[0]
    samples = np.concatenate((target, prediction, abs(target - prediction)), axis=0)
    # print(samples.shape)
    interp = None
    if vmax is None:
        vmin, vmax = [], []
        for i in range(n_fields):
            vmin.append(np.amin(samples[[i, i+n_fields]]))
            vmax.append(np.amax(samples[[i, i+n_fields]]))

    fig, axes = plt.subplots(3, n_fields, figsize=(3.5 * n_fields, 9))
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        if j < 2 * n_fields:
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap=cmap,
                                  vmin=vmin[j % n_fields], vmax=vmax[j % n_fields])
            elif plot_fn =='imshow':
                cax = ax.imshow(samples[j], cmap=cmap, origin='upper',
                                interpolation=interp,
                                vmin=vmin[j % n_fields], vmax=vmax[j % n_fields])   
        else:
            if same_scale:
                vmin_error, vmax_error = vmin[j % n_fields], vmax[j % n_fields]
            else:
                vmin_error, vmax_error = None, None
                if vmax_err is not None:
                    vmin_error, vmax_error = vmin_err[j % n_fields], vmax_err[j % n_fields]

            # if j == 8:
            #     vmin_error = None
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap=cmap)
            elif plot_fn =='imshow':
                cax = ax.imshow(samples[j], cmap=cmap, origin='upper',
                                interpolation=interp, vmin=vmin_error, vmax=vmax_error)
        if plot_fn == 'contourf':
            for c in cax.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.ax.yaxis.set_offset_position('left')
        # cbar.ax.tick_params(labelsize=5)
        cbar.update_ticks()
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=90)
    # plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    plt.subplots_adjust(top=0.93)
    plt.savefig(save_dir + '/pred_{}_{}.{}'.format(index, i_plot, ext),
                dpi=dpi, bbox_inches='tight')

    # plt.savefig(save_dir + '/pred_epoch{}_{}.{}'.format(epoch, index, ext),
    #             dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_prediction_bayes2(save_dir, target, pred_mean, pred_var, epoch, index, 
                        plot_fn='imshow', cmap='jet', same_scale=False):
    """Plot prediction for one input (`index`-th at epoch `epoch`)
    Args:
        save_dir: directory to save predictions
        target (np.ndarray): (3, 65, 65)
        prediction (np.ndarray): (3, 65, 65)
        epoch (int): which epoch
        index (int): i-th prediction
        plot_fn (str): choices=['contourf', 'imshow']
    """
    target, pred_mean, pred_std = to_numpy(target), to_numpy(pred_mean), np.sqrt(to_numpy(pred_var))
    
    rows = ['Simulation', 'Pred Mean', 'Pred Std', r'Sim $-$ Pred Mean']
    cols = ['Pressure', 'Horizontal Flux', 'Vertical Flux']

    # 3 x 65 x 65
    n_fields = target.shape[0]
    # 4, 3, 65, 65
    samples = np.stack((target, pred_mean, pred_std, target - pred_mean), axis=0)
    nrows = samples.shape[0]
    # print(samples.shape)
    interp = None
    vmin, vmax = [], []
    for j in range(n_fields):
        vmin.append(np.amin(samples[[0, 1], j]))
        vmax.append(np.amax(samples[[0, 1], j]))
        # vmin.append(np.amin(samples[[i, i+n_fields]]))
        # vmax.append(np.amax(samples[[i, i+n_fields]]))
    fig, axes = plt.subplots(samples.shape[0], n_fields, figsize=(3.75 * n_fields, 3 * nrows))
    for i in range(nrows):
        for j in range(n_fields):
            ax = axes[i, j]
    # for j, ax in enumerate(fig.axes):
            ax.set_aspect('equal')
            # ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            if i < 2:
                if plot_fn == 'contourf':
                    cax = ax.contourf(samples[i, j], 50, cmap=cmap,
                                    vmin=vmin[j], vmax=vmax[j])
                elif plot_fn =='imshow':
                    cax = ax.imshow(samples[i, j], cmap=cmap, origin='upper',
                                    interpolation=interp,
                                    vmin=vmin[j], vmax=vmax[j])   
            else:
                if same_scale:
                    vmin_error, vmax_error = vmin[j], vmax[j]
                else:
                    vmin_error, vmax_error = None, None
                if plot_fn == 'contourf':
                    cax = ax.contourf(samples[i, j], 50, cmap=cmap)
                elif plot_fn =='imshow':
                    cax = ax.imshow(samples[i, j], cmap=cmap, origin='upper',
                                    interpolation=interp, vmin=vmin_error, vmax=vmax_error)
            if plot_fn == 'contourf':
                for c in cax.collections:
                    c.set_edgecolor("face")
                    c.set_linewidth(0.000000000001)
            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.ax.yaxis.set_offset_position('left')
            # cbar.ax.tick_params(labelsize=5)
            cbar.update_ticks()
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, size='large')

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    # plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    # plt.subplots_adjust(top=0.93)
    plt.savefig(save_dir + '/pred_epoch{}_{}.{}'.format(epoch, index, ext),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_stats(save_dir, logger, *metrics):
    for metric in metrics:
        metric_list = logger[metric]
        np.savetxt(save_dir + f'/{metric}.txt', metric_list)
        # plot stats
        metric_arr = np.loadtxt(save_dir + f'/{metric}.txt')
        if len(metric_arr.shape) == 1:
            metric_arr = metric_arr[:, None]
        lines = plt.plot(range(1, len(metric_arr)+1), metric_arr)
        labels = [f'{metric_arr[-5:, i].mean():.4f}' for i in range(metric_arr.shape[-1])]
        plt.legend(lines, labels)
        plt.savefig(save_dir + f'/{metric}.pdf')
        plt.close()


def plot_prediction_bayes(save_dir, target, pred_mean, pred_var, epoch, index, 
        plot_fn='contourf'):
    """Plot predictions at *one* test input
    Args:
        save_dir: directory to save predictions
        target (np.ndarray or torch.Tensor): (3, 65, 65)
        pred_mean (np.ndarray or torch.Tensor): (3, 65, 65)
        pred_var (np.ndarray or torch.Tensor): (3, 65, 65)
        epoch (int): which epoch
        index (int): i-th prediction
        plot_fn (str): choices=['contourf', 'imshow']
    """
    target, pred_mean, pred_var = to_numpy(target), to_numpy(pred_mean), to_numpy(pred_var)

    pred_error = target - pred_mean    
    two_sigma = np.sqrt(pred_var) * 2
    # target: C x H x W
    sfmt = ticker.ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))
    cmap = 'jet'
    interpolation = None
    fig = plt.figure(1, (11, 12))
    axes_pad = 0.25
    cbar_pad = 0.1
    label_size = 6

    subplots_position = ['23{}'.format(i) for i in range(1, 7)]

    for i, subplot_i in enumerate(subplots_position):
        if i < 3:
            # share one colorbar
            grid = ImageGrid(fig, subplot_i,          # as in plt.subplot(111)
                             nrows_ncols=(2, 1),
                             axes_pad=axes_pad,
                             share_all=False,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="3%",
                             cbar_pad=cbar_pad,
                             )
            data = (target[i], pred_mean[i])
            channel = np.concatenate(data)
            vmin, vmax = np.amin(channel), np.amax(channel)
            # Add data to image grid
            for j, ax in enumerate(grid):
                if plot_fn == 'contourf':
                    im = ax.contourf(data[j], 50, vmin=vmin, vmax=vmax, cmap=cmap)
                    for c in im.collections:
                        c.set_edgecolor("face")
                        c.set_linewidth(0.000000000001)
                elif plot_fn == 'imshow':
                    im = ax.imshow(data[j], vmin=vmin, vmax=vmax,
                        interpolation=interpolation, cmap=cmap)
                ax.set_axis_off()
            # ticks=np.linspace(vmin, vmax, 10)
            #set_ticks, set_ticklabels
            cbar = grid.cbar_axes[0].colorbar(im, format=sfmt)
            # cbar.ax.set_yticks((vmin, vmax))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.ax.tick_params(labelsize=label_size)
            cbar.ax.toggle_label(True)

        else:
            grid = ImageGrid(fig, subplot_i,  # as in plt.subplot(111)
                             nrows_ncols=(2, 1),
                             axes_pad=axes_pad,
                             share_all=False,
                             cbar_location="right",
                             cbar_mode="each",
                             cbar_size="6%",
                             cbar_pad=cbar_pad,
                             )
            data = (pred_error[i-3], two_sigma[i-3])
            # channel = np.concatenate(data)
            # vmin, vmax = np.amin(channel), np.amax(channel)
            # Add data to image grid
            for j, ax in enumerate(grid):
                if plot_fn == 'contourf':
                    im = ax.contourf(data[j], 50, cmap=cmap)
                    for c in im.collections:
                        c.set_edgecolor("face")
                        c.set_linewidth(0.000000000001)
                elif plot_fn == 'imshow':
                    im = ax.imshow(data[j], interpolation=interpolation, cmap=cmap)
                ax.set_axis_off()
                cbar = grid.cbar_axes[j].colorbar(im, format=sfmt)
                grid.cbar_axes[j].tick_params(labelsize=label_size)
                grid.cbar_axes[j].toggle_label(True)
                # cbar.formatter.set_powerlimits((0, 0))
                cbar.ax.yaxis.set_offset_position('left')
                # print(dir(cbar.ax.yaxis))
                # cbar.update_ticks()

    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    fig.subplots_adjust(wspace=0.075, hspace=0.075)

    plt.savefig(save_dir + '/pred_at_x_epoch{}_{}.{}'.format(epoch, index, ext), 
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_MC(save_dir, monte_carlo, pred_mean, pred_var, mean, n_train):
    """Plot Monte Carlo Output
    
    Args:
        monte_carlo (np.ndarray or torch.Tensor): simulation output
        pred_mean (np.ndarray or torch.Tensor): from surrogate
        pred_var (np.ndarray or torch.Tensor): predictive var using surrogate
        mean (bool): Used in printing. True for plotting mean, False for var
    """
    monte_carlo, pred_mean, pred_var = to_numpy(monte_carlo), \
                                       to_numpy(pred_mean), to_numpy(pred_var)

    two_sigma = 2 * np.sqrt(pred_var)
    # target: C x H x W
    sfmt = ticker.ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((0, 0))
    cmap = 'jet'
    interpolation = 'bilinear'
    pred_error = monte_carlo - pred_mean
    fig = plt.figure(1, (10, 10))
    axes_pad = 0.25
    cbar_pad = 0.1
    label_size = 6

    subplots_position = ['23{}'.format(i) for i in range(1, 7)]

    for i, subplot_i in enumerate(subplots_position):
        if i < 3:
            # share one colorbar
            grid = ImageGrid(fig, subplot_i,          # as in plt.subplot(111)
                             nrows_ncols=(2, 1),
                             axes_pad=axes_pad,
                             share_all=False,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="3%",
                             cbar_pad=cbar_pad,
                             )
            data = (monte_carlo[i], pred_mean[i])
            channel = np.concatenate(data)
            vmin, vmax = np.amin(channel), np.amax(channel)
            # Add data to image grid
            for j, ax in enumerate(grid):
                # im = ax.imshow(data[j], vmin=vmin, vmax=vmax,
                #                interpolation=interpolation, cmap=cmap)
                im = ax.contourf(data[j], 50, vmin=vmin, vmax=vmax, cmap=cmap)
                for c in im.collections:
                    c.set_edgecolor("face")
                    c.set_linewidth(0.000000000001)
                ax.set_axis_off()
            # ticks=np.linspace(vmin, vmax, 10)
            #set_ticks, set_ticklabels
            cbar = grid.cbar_axes[0].colorbar(im, format=sfmt)
            # cbar.ax.set_yticks((vmin, vmax))
            cbar.ax.tick_params(labelsize=label_size)
            cbar.ax.yaxis.set_offset_position('left')
            cbar.ax.toggle_label(True)

        else:
            grid = ImageGrid(fig, subplot_i,  # as in plt.subplot(111)
                             nrows_ncols=(2, 1),
                             axes_pad=axes_pad,
                             share_all=False,
                             cbar_location="right",
                             cbar_mode="each",
                             cbar_size="6%",
                             cbar_pad=cbar_pad,
                             )
            data = (pred_error[i-3], two_sigma[i-3])
            # channel = np.concatenate(data)
            # vmin, vmax = np.amin(channel), np.amax(channel)
            # Add data to image grid
            for j, ax in enumerate(grid):
                # im = ax.imshow(data[j], interpolation=interpolation, cmap=cmap)
                im = ax.contourf(data[j], 50, cmap=cmap)
                for c in im.collections:
                    c.set_edgecolor("face")
                    c.set_linewidth(0.000000000001)
                ax.set_axis_off()
                cbar = grid.cbar_axes[j].colorbar(im, format=sfmt)
                grid.cbar_axes[j].tick_params(labelsize=label_size)
                grid.cbar_axes[j].toggle_label(True)
                # cbar.formatter.set_powerlimits((0, 0))
                cbar.ax.yaxis.set_offset_position('left')
                # print(dir(cbar.ax.yaxis))
                # cbar.update_ticks()

    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    fig.subplots_adjust(wspace=0.075, hspace=0.075)

    plt.savefig(save_dir + '/pred_{}_vs_MC.pdf'.format('mean' if mean else 'var'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Done plotting Pred_{}_vs_MC, num of training: {}"
          .format('mean' if mean else 'var', n_train))



def plot_MC2(save_dir, monte_carlo, pred_mean, pred_var, mean, ntrain,
    plot_fn='imshow', cmap='jet', manual_scale=False, same_scale=False):
    """Plot Monte Carlo Output
    
    Args:
        monte_carlo (np.ndarray or torch.Tensor): simulation output
        pred_mean (np.ndarray or torch.Tensor): from surrogate
        pred_var (np.ndarray or torch.Tensor): predictive var using surrogate
        mean (bool): Used in printing. True for plotting mean, False for var
    """

    target, pred_mean, pred_std = to_numpy(monte_carlo), to_numpy(pred_mean), np.sqrt(to_numpy(pred_var))
    
    if mean:
        rows = ['Monte Carlo', 'Mean of Est. Mean', '2 Std of Est. Mean', 'Row1 - Row2']
    else:
        rows = ['Monte Carlo', 'Mean of Est. Variance', '2 Std of Est. Variance', 'Row1 - Row2']
    cols = ['Pressure', 'Horizontal Flux', 'Vertical Flux']

    # 3 x 65 x 65
    n_fields = target.shape[0]
    # 4, 3, 65, 65
    samples = np.stack((target, pred_mean, pred_std * 2, target - pred_mean), axis=0)
    nrows = samples.shape[0]
    # print(samples.shape)
    interp = None
    vmin, vmax = [], []
    for j in range(n_fields):
        vmin.append(np.amin(samples[[0, 1], j]))
        vmax.append(np.amax(samples[[0, 1], j]))

    # manually set the vmin and vmax
    if manual_scale and mean:
        vmin[1], vmax[1] = 1.0, 1.1
        # vmin[2], vmax[2] = -0.05, 0.05
         
    fig, axes = plt.subplots(samples.shape[0], n_fields, figsize=(3.75 * n_fields, 3 * nrows))
    for i in range(nrows):
        for j in range(n_fields):
            ax = axes[i, j]
    # for j, ax in enumerate(fig.axes):
            ax.set_aspect('equal')
            # ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            if i < 2:
                if plot_fn == 'contourf':
                    cax = ax.contourf(samples[i, j], 50, cmap=cmap,
                                    vmin=vmin[j], vmax=vmax[j])
                elif plot_fn =='imshow':
                    cax = ax.imshow(samples[i, j], cmap=cmap, origin='upper',
                                    interpolation=interp,
                                    vmin=vmin[j], vmax=vmax[j])   
            else:
                if same_scale:
                    vmin_error, vmax_error = vmin[j], vmax[j]
                else:
                    vmin_error, vmax_error = None, None
                if plot_fn == 'contourf':
                    cax = ax.contourf(samples[i, j], 50, cmap=cmap)
                elif plot_fn =='imshow':
                    cax = ax.imshow(samples[i, j], cmap=cmap, origin='upper',
                                    interpolation=interp, vmin=vmin_error, vmax=vmax_error)
            if plot_fn == 'contourf':
                for c in cax.collections:
                    c.set_edgecolor("face")
                    c.set_linewidth(0.000000000001)
            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.ax.yaxis.set_offset_position('left')
            # cbar.ax.tick_params(labelsize=5)
            cbar.update_ticks()
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, size='large')

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    # plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    # plt.subplots_adjust(top=0.93)
    plt.savefig(save_dir + '/pred_{}_vs_MC.pdf'.format('mean' if mean else 'var'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Done plotting Pred_{}_vs_MC, num of training: {}"
          .format('mean' if mean else 'var', ntrain))



def plot_UP(save_dir, monte_carlo, surr_mean, is_mean,
        plot_fn='imshow', cmap='jet', same_scale=False):
    """Plot uncertainty propagation, for deep ensembles. Only mean estimate,
    no variance for each estimate.

    Args:
        save_dir: directory to save predictions
        target (np.ndarray): (3, 65, 65)
        prediction (np.ndarray): (3, 65, 65)
        epoch (int): which epoch
        index (int): i-th prediction
        plot_fn (str): choices=['contourf', 'imshow']
    """
    target, prediction = to_numpy(monte_carlo), to_numpy(surr_mean)

    rows = ['Simulator', 'Surrogate', r'Row1 $-$ Row2']
    cols = ['Pressure', 'Horizontal Flux', 'Vertical Flux']

    # 3 x 65 x 65
    n_fields = target.shape[0]
    samples = np.concatenate((target, prediction, target - prediction), axis=0)
    # print(samples.shape)
    interp = None
    vmin, vmax = [], []
    for i in range(n_fields):
        vmin.append(np.amin(samples[[i, i+n_fields]]))
        vmax.append(np.amax(samples[[i, i+n_fields]]))

    fig, axes = plt.subplots(3, n_fields, figsize=(3.75 * n_fields, 9))
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        if j < 2 * n_fields:
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap=cmap,
                                  vmin=vmin[j % n_fields], vmax=vmax[j % n_fields])
            elif plot_fn =='imshow':
                cax = ax.imshow(samples[j], cmap=cmap, origin='upper',
                                interpolation=interp,
                                vmin=vmin[j % n_fields], vmax=vmax[j % n_fields])   
        else:
            if same_scale:
                vmin_error, vmax_error = vmin[j % n_fields], vmax[j % n_fields]
            else:
                vmin_error, vmax_error = None, None
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap=cmap)
            elif plot_fn =='imshow':
                cax = ax.imshow(samples[j], cmap=cmap, origin='upper',
                                interpolation=interp, vmin=vmin_error, vmax=vmax_error)
        if plot_fn == 'contourf':
            for c in cax.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.ax.yaxis.set_offset_position('left')
        # cbar.ax.tick_params(labelsize=5)
        cbar.update_ticks()
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, size='large')

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    # plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    # plt.subplots_adjust(top=0.93)
    plt.savefig(save_dir + '/pred_{}_vs_MC.pdf'.format('mean' if is_mean else 'var'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Done plotting Pred_{}_vs_MC".format('mean' if is_mean else 'var'))
    

def save_samples(save_dir, images, epoch, index, name, nrow=4, heatmap=True, cmap='jet', title=False):
    """Save samples in grid as images or plots
    Args:
        images (Tensor): B x C x H x W
    """

    # if images.shape[0] < 10:
    #     nrow = 2
    #     ncol = images.shape[0] // nrow
    # else:
    #     ncol = nrow
    images = to_numpy(images)
    ncol = images.shape[0] // nrow

    if heatmap:
        for c in range(images.shape[1]):
            # (11, 12)
            fig = plt.figure(1, (12, 12))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(nrow, ncol),
                             axes_pad=0.1,
                             share_all=False,
                             cbar_location="top",
                             cbar_mode="single",
                             cbar_size="3%",
                             cbar_pad=0.1
                             )
            for j, ax in enumerate(grid):
                im = ax.imshow(images[j][c], cmap=cmap)
                ax.set_axis_off()
                ax.set_aspect('equal')
            cbar = grid.cbar_axes[0].colorbar(im)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.toggle_label(True)
            # change plot back to epoch
            if title:
                plt.suptitle(f'Epoch {epoch}')
                plt.subplots_adjust(top=0.95)
            plt.savefig(save_dir + '/epoch{}_{}_c{}_index{}.png'.format(epoch, name, c, index),
                        bbox_inches='tight')
            plt.close(fig)
    else:
        torchvision.utils.save_image(images, 
                          save_dir + '/fake_samples_epoch_{}.png'.format(epoch),
                          nrow=nrow,
                          normalize=True)


def plot_row(arrs, save_dir, filename, same_range=False, plot_fn='imshow', 
    cmap='viridis'):
    """
    Args:
        arrs (sequence of 2D Tensor or Numpy): seq of arrs to be plotted
        save_dir (str):
        filename (str):
        same_range (bool): if True, subplots have the same range (colorbar)
        plot_fn (str): choices=['imshow', 'contourf']
    """
    interpolation = None
    arrs = [to_numpy(arr) for arr in arrs]

    if same_range:
        vmax = max([np.amax(arr) for arr in arrs])
        vmin = min([np.amin(arr) for arr in arrs])
    else:
        vmax, vmin = None, None

    fig, _ = plt.subplots(1, len(arrs), figsize=(4.4 * len(arrs), 4))
    for i, ax in enumerate(fig.axes):
        if plot_fn == 'imshow':
            cax = ax.imshow(arrs[i], cmap=cmap, interpolation=interpolation,
                            vmin=vmin, vmax=vmax)
        elif plot_fn == 'contourf':
            cax = ax.contourf(arrs[i], 50, cmap=cmap, vmin=vmin, vmax=vmax)
        if plot_fn == 'contourf':
            for c in cax.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
        ax.set_axis_off()
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.ax.yaxis.set_offset_position('left')
        # cbar.ax.tick_params(labelsize=5)
        cbar.update_ticks()
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    plt.savefig(save_dir + f'/{filename}.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
