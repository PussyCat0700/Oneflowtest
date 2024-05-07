import torch
import oneflow as flow
from tqdm import tqdm
from utils import SeparateWriter, Timer, oneflow_to_global
from thop import profile
from flowflops import get_model_complexity_info
from config import cfgs
from generate_model import generate_model

def compare_models(writer:SeparateWriter, num_updates=1, model_name="ResNet50"):
    BATCH_SIZE = cfgs[model_name]['BATCH_SIZE']
    INPUT_SHAPE = cfgs[model_name]['INPUT_SHAPE']
    NUM_CLASSES = cfgs[model_name]["NUM_CLASSES"]
    VOCAB_SIZE = cfgs[model_name]["VOCAB_SIZE"]

    # 构建模型并放到GPU上
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def _get_input_tensor():
        # 输入和目标
        tinput = torch.randint(0, VOCAB_SIZE, INPUT_SHAPE, device=device)
        tattn_mask = torch.ones_like(tinput)
        # 对于Oneflow的BERT-Large，不包上oneflow_to_global跑不了。
        finput = oneflow_to_global(flow.tensor(tinput.cpu()))
        fattn_mask = oneflow_to_global(flow.BoolTensor(tattn_mask.cpu()))
        return tinput, tattn_mask, finput, fattn_mask
    tmodel, fmodel = generate_model(model_name)
    tmodel = tmodel.to(device)
    fmodel = fmodel.to(device)
    timer_t = Timer()
    timer_f = Timer()

    # 拷贝参数放在模型生成里面，一些模型的参数对不上，要手动对齐。
    # state_dict = {k: v.cpu().numpy() for k, v in tmodel.state_dict().items()}
    # fmodel.load_state_dict({k: flow.tensor(v) for k, v in state_dict.items()})

    # 损失函数和优化器
    lr = 1e-3
    tloss_fn = torch.nn.CrossEntropyLoss()
    floss_fn = flow.nn.CrossEntropyLoss()
    toptimizer = torch.optim.Adam(tmodel.parameters(), lr=lr)
    foptimizer = flow.optim.Adam(fmodel.parameters(), lr=lr)
    pbar = tqdm(range(num_updates))
    for step in pbar:
        tinput, tattn_mask, finput, fattn_mask = _get_input_tensor()
        tgt = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, NUM_CLASSES), device=device)
        tgt_flow = oneflow_to_global(flow.tensor(tgt.cpu().numpy()).to(device))

        # 测量前向传播时间
        torch.cuda.synchronize()
        with timer_t.time():
            tout = tmodel(tinput, tattn_mask)
            torch.cuda.synchronize()
            torch_time = timer_t.get_and_reset()

        flow.cuda.synchronize()
        with timer_f.time():
            fout = fmodel(finput, fattn_mask)
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
        # writer.write_log('Troughput (Sample per second)', BATCH_SIZE/torch_time, BATCH_SIZE/flow_time, step)
        writer.write_log('Backward Time', torch_backward_time, flow_backward_time, step)
    
    # FLOPs和参数量计算
    ## TODO 使用get_model_complexity_info会报错
    ## Error: RuntimeError : Only local tensors have 'device'. Please use '.placement' for global tensors.
    # tinput, tattn_mask, _, _ = _get_input_tensor()
    # tflops, tparams = profile(model=tmodel, inputs=(tinput, tattn_mask))
    # writer.write_log_single('Torch/FLOPs', tflops, None)
    # writer.write_log_single('Torch/Params', tparams, None)
    
    # # 关于Eager和Graph：https://github.com/Oneflow-Inc/flow-OpCounter/blob/master/README_CN.md
    # for mode in ["eager", "graph"]:
    #     if model_name == "SwinTransformer" and mode == 'graph':
    #         continue
    #     fflops, fparams = get_model_complexity_info(
    #         fmodel, INPUT_SHAPE,
    #         as_strings=False,
    #         print_per_layer_stat=False,
    #         mode=mode
    #     )
    #     writer.write_log_single(f'OneFlow/FLOPs[{mode}]', fflops, None)
        # writer.write_log_single(f'OneFlow/Params[{mode}]', fparams, None)
