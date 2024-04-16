import torch
import oneflow as flow
import numpy as np
import random
from single_step import compare_models
from torch.utils.tensorboard import SummaryWriter
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare_updates", type=int, default=1000)
    parser.add_argument("--run_name", default="run")
    return parser.parse_args()    

run_name = "model_comparison"
writer = SummaryWriter(f'runs/{run_name}')
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
    print('model comparison')
    compare_models(writer=writer, run_name=args.run_name, num_updates=args.compare_updates)