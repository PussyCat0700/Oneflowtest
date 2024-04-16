import torch
import oneflow as flow
import torchvision
import flowvision
import numpy as np
import random

# 固定种子
seed = 123456
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
flow.manual_seed(seed)
flow.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True

# 构建模型
tmodel = torchvision.models.resnet50()
fmodel = flowvision.models.resnet50()

# 拷贝参数
state_dict = {k:v.numpy() for k,v in tmodel.state_dict().items()}  # must be numpied in oneflow v0.9.0
fmodel.load_state_dict(state_dict) # <All keys matched successfully>

# 损失函数和优化器
lr = 1e-3
tloss_fn = torch.nn.CrossEntropyLoss()
floss_fn = flow.nn.CrossEntropyLoss()
toptimizer = torch.optim.Adam(tmodel.parameters(), lr=lr)
foptimizer = flow.optim.Adam(fmodel.parameters(), lr=lr)

# 输入
tinput = torch.rand((4, 3, 256, 256))
finput = flow.tensor(tinput)
tgt = torch.rand((4,1000))

# 运行
tout = tmodel(tinput)
fout = fmodel(finput)

# 输出结果比较
tout_data = flow.tensor(tout.data)
fout_data = fout.data
fout_data.equal(tout_data) # True

# loss比较
tloss = tloss_fn(tout, tgt)
floss = floss_fn(fout, flow.tensor(tgt))
tloss.item() == floss.item() # True

# 梯度比较
toptimizer.zero_grad()
tloss.backward()
foptimizer.zero_grad()
floss.backward()
tgrad = [param for param in tmodel.parameters() if param.requires_grad and param.grad is not None]
fgrad = [param for param in fmodel.parameters() if param.requires_grad and param.grad is not None]
assert len(tgrad)==len(fgrad)
for i in range(len(tgrad)):
    fgrad[i].data.equal(flow.tensor(tgrad[i].data)) # True

# 更新后的参数比较
toptimizer.step()
foptimizer.step()
tstate_dict = tmodel.state_dict()
fstate_dict = fmodel.state_dict()
for key,value in tstate_dict.items():
    fstate_dict[key].equal(flow.tensor(value)) # True
