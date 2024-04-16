import time
import numpy as np
import torch
import oneflow
from torch.utils.tensorboard import SummaryWriter

class Timer:
    def __init__(self):
        self.start_time = None
        self.total_time = 0

    def time(self):
        return self

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.total_time += elapsed_time

    def get_and_reset(self):
        total = self.total_time
        self.total_time = 0
        return total
    
class SeparateWriter:
    BOTH = "both"
    ONEFLOW_ONLY = "oneflow_only"
    TORCH_ONLY = "torch_only"
    LOGMODES = [BOTH, ONEFLOW_ONLY, TORCH_ONLY]  # only valid for single
    def __init__(self, run_name) -> None:
        self.writer = SummaryWriter(f'runs/{run_name}')
        self.logmode = self.BOTH
    
    def write_log(self, logstr, torch_data, oneflow_data, steps):
        self.writer.add_scalars(logstr, {'PyTorch': torch_data, 'OneFlow': oneflow_data}, steps)
        
    def write_log_single(self, logstr, data, steps):
        if self.logmode == self.TORCH_ONLY:
            key = "PyTorch"
            if isinstance(data, torch.Tensor):
                data = data.item()
        else:
            key = "OneFlow"
            if isinstance(data, oneflow.Tensor):
                data = data.numpy()
        self.writer.add_scalars(logstr, {key: data}, steps)
        
    def set_log_mode(self, mode):
        if mode not in self.LOGMODES:
            raise NotImplementedError(f"got {mode=} but should be one of {self.LOGMODES}")
        self.logmode = mode
        
    def see_diff_tensor(self, logstr, torch_data:np.ndarray, oneflow_data:np.ndarray, steps):
        multi_dims = isinstance(torch_data, np.ndarray)
        
        if multi_dims:
            # 计算均值差异
            mean_difference = (torch_data - oneflow_data).mean()

            # 计算标准偏差差异
            std_deviation_difference = torch_data.std() - oneflow_data.std()

            # 计算最大偏差
            max_difference = np.abs(torch_data - oneflow_data).max()
            self.writer.add_scalar(logstr+"△mean", mean_difference, steps)
            self.writer.add_scalar(logstr+"△std", std_deviation_difference, steps)
            self.writer.add_scalar(logstr+"△maxdiff", max_difference, steps)
        else:
            # 计算相对偏差百分比
            relative_difference = (np.abs(torch_data - oneflow_data) / np.abs(torch_data).clip(min=1e-6)).mean() * 100
            self.writer.add_scalar(logstr+"△percent", relative_difference, steps)

if __name__ == '__main__':
    from time import sleep
    timer_1 = Timer()

    with timer_1.time():
        sleep(1)

    # Block 2: This block will not be timed
    sleep(1)

    with timer_1.time():
        # Block 3: This block will also be timed and its time will add to the timer
        sleep(1)

    # Block 4: This block will not be timed
    sleep(1)

    total_time_timer_1 = timer_1.get_and_reset()
    print(f"Total timed duration: {total_time_timer_1} seconds")  # 2 seconds
