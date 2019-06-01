"""
Post processing, mainly for uncertainty quantification tasks using pre-trained
conditional Glow.
"""
import torch
from models.glow_msc import MultiScaleCondGlow
from utils.uq import UQ_CondGlow
from utils.load import load_args
from utils.misc import mkdirs
from torch.utils.data import DataLoader, TensorDataset
from time import time
import h5py
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=str, default=None, help='directory to an experiment run')
parser.add_argument('--n-pred', type=int, default=32, help='# prediction at x')
parser.add_argument('--n-samples', type=int, default=20, help='# predictive output samples per input')
parser.add_argument('--plot-samples', action='store_true', default=False, help='plot predictive output samples per input')
parser.add_argument('--var-samples', type=int, default=10, 
    help='# exprs to evaluate the mean and var of the statistics in uncertainty propagation')
parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature for Gaussian noise')
parser.add_argument('--n-loc', type=int, default=40, help='# locations on which to evalute the PDF')
parser.add_argument('--n-mc', type=int, default=10000, help='# data for Monte Carlo')
parser.add_argument('--mc-bs', type=int, default=1000, help='batch size for Monte Carlo')
parser.add_argument('--cuda', type=int, default=0, help='gpu #')
args_post = parser.parse_args()

run_dir = './experiments/cglow/reverse_kld/kle100_ntrain4096_ENC_blocks[3, 4, 4]_'\
    'FLOW_blocks[6, 6, 6]_wb50_beta150.0_batch32_lr0.0015_epochs400'

if args_post.run_dir is not None:
    run_dir = args_post.run_dir

device = torch.device(f'cuda:{args_post.cuda}' if torch.cuda.is_available() else 'cpu')
# load the args for pre-trained run
args = load_args(run_dir)
args.device = device
args.post_dir = args.run_dir + f'/post_proc_mc{args_post.n_mc}_nsamples{args_post.n_samples}_'\
    f'varsamples{args_post.var_samples}_temp{args_post.temperature}'
mkdirs(args.post_dir)

# load the pre-trained model
cglow = MultiScaleCondGlow(img_size=args.imsize, 
                        x_channels=args.x_channels, 
                        y_channels=args.y_channels, 
                        enc_blocks=args.enc_blocks, 
                        flow_blocks=args.flow_blocks, 
                        LUdecompose=args.LU_decompose,
                        squeeze_factor=2,
                        data_init=args.data_init)
print(cglow.model_size)
ckpt_file = args.ckpt_dir + f"/model_epoch{args.epochs}.pth"
checkpoint = torch.load(ckpt_file, map_location='cpu')
cglow.load_state_dict(checkpoint['model_state_dict'])
cglow = cglow.to(device)
if args.data_init:
    cglow.init_actnorm()
cglow.eval()
print(f'Loaded checkpoint: {ckpt_file}')

# load Monte Carlo data: 10,000
hdf5_file = args.data_dir + f'/{args.imsize}x{args.imsize}/kle{args.kle}_lhs10000_monte_carlo.hdf5'
tic = time()
with h5py.File(hdf5_file, 'r') as f:
    x_data = f['input'][-args_post.n_mc:]
    y_data = f['output'][-args_post.n_mc:]
    print(f'y: {y_data.shape}')
mc_loader = DataLoader(TensorDataset(torch.FloatTensor(x_data), 
    torch.FloatTensor(y_data)),
    batch_size=args_post.mc_bs, shuffle=False, drop_last=True)
print(f'Loaded Monte Carlo data in {hdf5_file} in {time()-tic} seconds')

# load test data
if args.kle == 100:
    test_hdf5_file = args.data_dir + f'/{args.imsize}x{args.imsize}/kle{args.kle}_lhs1000_test.hdf5'
with h5py.File(test_hdf5_file, 'r') as f:
    x_test = f['input'][:args.ntest]
    # no sim data yet
    y_test = f['output'][:args.ntest]
    print(f'y_test: {y_test.shape}')
y_test_variation = ((y_test - y_test.mean(0, keepdims=True)) ** 2).sum(axis=(0, 2, 3))
test_loader = DataLoader(TensorDataset(torch.FloatTensor(x_test), 
    torch.FloatTensor(y_test)),
    batch_size=512, shuffle=True, drop_last=True)
print(f'Loaded test data in: {test_hdf5_file}')

# Now performs UQ tasks
uq = UQ_CondGlow(cglow, args, mc_loader, test_loader, y_test_variation, 
    n_samples=args_post.n_samples, temperature=args_post.temperature)

with torch.no_grad():
    uq.plot_prediction_at_x(n_pred=args_post.n_pred, plot_samples=args_post.plot_samples)
    uq.plot_dist(num_loc=args_post.n_loc)
    uq.test_metric(handle_nan=True)
    uq.plot_reliability_diagram(save_time=True)
    uq.propagate_uncertainty(manual_scale=False, var_samples=args_post.var_samples)
