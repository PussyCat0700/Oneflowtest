1. 程序入口main.py
1. --wandb 正式跑的时候打开，留个档便于交付和展示。*调试不用打开这个选项*
1. --run_name 替换为模型名（如ResNet50）便于归档。wandb的project名和本地tensorboard的run name。*调试不用打开这个选项*
1. --model 选择需要跑的模型（ResNet50，Inception, SEResNet50, MobileNet, ShuffleNet, DenseNet, SwinTransformer, EfficientNet），默认ResNet50
1. --stage 只是用来区分三个stage的
    - stage 1 (compare): 比较feature和loss的差异。这一步只会记录oneflow和torch的特征差异（所谓的正确性测试），跑1000 steps就停了
    - stage 2 (training)：正式在CIFAR-10上训练10个epoch（oneflow）。记录硬件占用和相关指标。
    - stage 3 (training)：正式在CIFAR-10上训练10个epoch（torch）。记录硬件占用和相关指标。
    - 后面需要加入更多模型测试的时候三个stage都要拉通跑一遍，可以删掉这个参数拉通三个stage全跑了

# Oneflow 安装
```bash
python3 -m pip install --upgrade pip #--user， pip 24.0
python3 -m pip install oneflow  # 0.9.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # torch 2.2.2+cu118
pip install flowvision==0.2.1
pip install tensorboard
pip install thop
pip install flowflops
pip install torchmetrics
pip install wandb
pip install scipy
```

# NLP模型测试
## T5和Bert-Large
### 数据集
使用conll2003作为数据集，详情见此[notebook示例](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ)
### 配置文件
见detailed_configs

## GPT2和LLaMa-7B
使用Alpaca dataset
