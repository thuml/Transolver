import os
import argparse
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
import logging
import pprint
import matplotlib.pyplot as plt
from tqdm import *
from utils_Dri import setup_logger
from einops import rearrange
from model_dict import get_model
from utils.normalizer import UnitTransformer
from utils.testloss import TestLoss
from datetime import datetime
from colorama import Fore, Style

#! alias for colorful output
R = Fore.RED
Y = Fore.YELLOW
G = Fore.GREEN
RESET = Style.RESET_ALL

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser('Training Translover')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default='Transolver_2D')
    parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
    parser.add_argument('--n-layers', type=int, default=3, help='layers')
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--gpu", type=str, default='1', help="GPU index to use")
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--downsample', type=int, default=5)
    parser.add_argument('--mlp_ratio', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--unified_pos', type=int, default=0)
    parser.add_argument('--ref', type=int, default=8)
    parser.add_argument('--slice_num', type=int, default=32)
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--save_name', type=str, default='darcy_Transolver')
    parser.add_argument('--data_path', type=str, default='/work/mae-zhangbj/Data_store/Data_Pressure_Darcy/')

    return parser.parse_args()


def main():

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = "Tran Test"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    setup_logger(log_file)
    logging.info(f"{Fore.RED}*************************Start pressure prediction in Darcy flow.{Style.RESET_ALL}")

    # Set up arguments
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
    test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
    ntrain = args.ntrain
    ntest = 200
    epochs = args.epochs
    eval = args.eval
    save_name = args.save_name


    # Downsampling and Setup the mesh grid
    # Prepare the train_data and test_data
    r  = args.downsample
    h  = int(((421 - 1) / r) + 1)
    s  = h
    dx = 1.0 / s

    train_data = scio.loadmat(train_path)
    x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
    x_train = x_train.reshape(ntrain, -1)
    x_train = torch.from_numpy(x_train).float()

    y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
    y_train = y_train.reshape(ntrain, -1)
    y_train = torch.from_numpy(y_train)

    test_data = scio.loadmat(test_path)
    x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
    x_test = x_test.reshape(ntest, -1)
    x_test = torch.from_numpy(x_test).float()

    y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
    y_test = y_test.reshape(ntest, -1)
    y_test = torch.from_numpy(y_test)

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    y_normalizer.cuda()

    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    logging.info(f"{G} pos.shape: {pos.shape}{RESET}")

    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    logging.info(f"Dataloading is over.")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                              batch_size=args.batch_size, shuffle=False)

    # Create the Transolver model
    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=1,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  H=s, W=s).cuda()

    # END
    logging.info(f"{Fore.RED}*******************************************The train is Done.{Style.RESET_ALL}")


if __name__=="__main__":
    exit(main())

