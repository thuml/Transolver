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

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
ntrain = args.ntrain
ntest = 200
epochs = args.epochs
eval = args.eval
save_name = args.save_name

def main():

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = "Tran Test"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    setup_logger(log_file)
    logging.info(f"Start pressure prediction in Darcy flow")

    r  = args.downsample
    h  = int(((421 - 1) / r) + 1)
    s  = h
    dx = 1.0 / s

    train_data = scio.loadmat(train_path)
    logging.info(f"Fore.YELLOW train_data:")

#    x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]







if __name__=="__main__":
    exit(main())

