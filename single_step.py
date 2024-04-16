import torch
import oneflow as flow
import torchvision
import flowvision
import numpy as np
import random
from tqdm import tqdm
from utils import SeparateWriter, Timer
def compare_models(writer:SeparateWriter, run_name, num_updates=1):
    # 构建模型并放到GPU上
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmodel = torchvision.models.resnet50().to(device)
    fmodel = flowvision.models.resnet50().to(device)
    timer_t = Timer()
    timer_f = Timer()


    # 拷贝参数
    state_dict = {k: v.cpu().numpy() for k, v in tmodel.state_dict().items()}
    fmodel.load_state_dict({k: flow.tensor(v) for k, v in state_dict.items()})

    # 损失函数和优化器
    lr = 1e-3
    tloss_fn = torch.nn.CrossEntropyLoss()
    floss_fn = flow.nn.CrossEntropyLoss()
    toptimizer = torch.optim.Adam(tmodel.parameters(), lr=lr)
    foptimizer = flow.optim.Adam(fmodel.parameters(), lr=lr)
    pbar = tqdm(range(num_updates))
    for step in pbar:
        # 输入和目标
        tinput = torch.rand((4, 3, 256, 256), device=device)
        finput = flow.tensor(tinput.cpu().numpy()).to(device)
        tgt = torch.randint(0, 1000, (4,), device=device)
        tgt_flow = flow.tensor(tgt.cpu().numpy()).to(device)

        # 测量前向传播时间
        torch.cuda.synchronize()
        with timer_t.time():
            tout = tmodel(tinput)
            torch.cuda.synchronize()
            torch_time = timer_t.get_and_reset()

        flow.cuda.synchronize()
        with timer_f.time():
            fout = fmodel(finput)
            flow.cuda.synchronize()
            flow_time = timer_f.get_and_reset()
        tout_data = tout.detach().cpu().numpy()
        fout_data = fout.numpy()
        writer.see_diff_tensor('Feature', tout_data, fout_data, step)

        # 计算loss
        tloss = tloss_fn(tout, tgt)
        floss = floss_fn(fout, tgt_flow)

        # 后向传播
        toptimizer.zero_grad()
        with timer_t.time():
            tloss.backward()
            torch.cuda.synchronize()
            torch_backward_time = timer_t.get_and_reset()

        foptimizer.zero_grad()
        with timer_f.time():
            floss.backward()
            flow.cuda.synchronize()
            flow_backward_time = timer_f.get_and_reset()

        # 更新参数
        toptimizer.step()
        foptimizer.step()

        # TensorBoard记录
        writer.write_log('Loss', tloss.item(), floss.numpy(), step)
        writer.see_diff_tensor('Loss', tloss.item(), floss.numpy(), step)
        writer.write_log('Forward Time', torch_time, flow_time, step)
        writer.write_log('Backward Time', torch_backward_time, flow_backward_time, step)
