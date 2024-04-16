import torch
import oneflow as flow
import numpy as np
import random
from single_step import compare_models
from multi_step import serious_train
from utils import SeparateWriter
import argparse
import wandb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare_updates", type=int, default=1000)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--run_name", default="run0")
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument('--stage', type=int, choices=[0,1,2], required=True)
    return parser.parse_args()    

# 固定种子
seed = 123456
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
flow.manual_seed(seed)
flow.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    args = get_args()
    enable_wandb = args.wandb
    if args.stage == 0:
        args.run_name += '_compare'
    if args.stage == 1:
        args.run_name += '_oneflow'
    elif args.stage == 2:
        args.run_name += '_torch'
    logging_dir = f'runs/{args.run_name}'
    if enable_wandb:
        wandb.tensorboard.patch(root_logdir=logging_dir)
        wandb.init(project=args.run_name)
    writer = SeparateWriter(logging_dir=logging_dir)
    if args.stage == 0:
        print('model comparison')
        compare_models(writer=writer, num_updates=args.compare_updates)
    elif args.stage == 1:
        print('Training OneFlow')
        serious_train(writer=writer, epochs=args.train_epochs, enable_oneflow=True)
    elif args.stage == 2:
        print('Training Torch')
        serious_train(writer=writer, epochs=args.train_epochs, enable_oneflow=False)