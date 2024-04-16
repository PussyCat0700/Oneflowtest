import torch
import oneflow as flow
import numpy as np
import random
from single_step import compare_models
from torch.utils.tensorboard import SummaryWriter
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
    compare_models(writer=writer, run_name=run_name)