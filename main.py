import torch
import oneflow as flow
import numpy as np
import random
from single_step import compare_models
from multi_step import serious_train
from utils import SeparateWriter
import argparse
import wandb
from config import cfgs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare_updates", type=int, default=1000)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--run_name", default="run0")
    parser.add_argument("--model", default='ResNet50', choices=cfgs['model_name'])
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
    if args.model not in cfgs['model_name']:
        print(f"Model {args.model} not implemented yet")
        raise NotImplementedError()
    enable_wandb = args.wandb
    if args.stage == 0:
        args.run_name += '_compare'
    if args.stage in [1, 2]:
        args.run_name += '_training'
    logging_dir = f'runs/{args.run_name}'
    if enable_wandb:
        wandb.tensorboard.patch(root_logdir=logging_dir)
        wandb.init(project=args.run_name, name='oneflow' if args.stage == 1 else 'torch')
    writer = SeparateWriter(logging_dir=logging_dir)
    if args.stage == 0:
        print('model comparison')
        compare_models(writer=writer, num_updates=args.compare_updates, model_name=args.model)
    elif args.stage == 1:
        print('Training OneFlow')
        serious_train(writer=writer, epochs=args.train_epochs, enable_oneflow=True, model_name=args.model)
    elif args.stage == 2:
        print('Training Torch')
        serious_train(writer=writer, epochs=args.train_epochs, enable_oneflow=False, model_name=args.model)