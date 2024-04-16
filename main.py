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
    logging_dir = f'runs/{args.run_name}'
    if enable_wandb:
        wandb.tensorboard.patch(root_logdir=logging_dir)
        wandb.init(project=args.run_name)
    writer = SeparateWriter(logging_dir=logging_dir)
    print('model comparison')
    compare_models(writer=writer, run_name=args.run_name, num_updates=args.compare_updates)
    print('Training OneFlow')
    serious_train(writer=writer, epochs=args.train_epochs, enable_oneflow=True)
    # TODO This can be run in a separate process in parallel to oneflow.
    print('Training Torch')
    serious_train(writer=writer, epochs=args.train_epochs, enable_oneflow=False)