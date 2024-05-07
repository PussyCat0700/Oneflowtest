import oneflow as flow
import torch
import torchmetrics
from utils import SeparateWriter
from multi_steps.detailed_configs.bert_large import cfg as bert_cfg
from generate_model import generate_model
from multi_steps.detailed_configs.bert_large import tokenized_datasets, data_collator, IGNORE_INDEX
# refer to https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ

def serious_train(writer:SeparateWriter, epochs:int, enable_oneflow:bool, model_name='ResNet50'):
    DEVICE = 'cuda' if flow.cuda.is_available() else 'cpu'

    BATCH_SIZE = 64
    NUM_CLASSES = bert_cfg["num_labels"]

    if enable_oneflow:
        import oneflow.nn as nn
        from oneflow.optim import SGD
        from oneflow.utils.data import DataLoader
        data_collator.enable_oneflow = True
    else:
        import torch.nn as nn
        from torch.optim import SGD
        from torch.utils.data import DataLoader

    train_data_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=data_collator)
    test_data_loader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=data_collator)
    # 如何保证两次跑出来的模型初始化参数一致？
    if enable_oneflow:
        _, model = generate_model(model_name)
    else:
        model, _ = generate_model(model_name)
    
    model = model.to(DEVICE)

    def evaluate(model, data_loader, steps, model_name):
        model.eval()
        num_classes, task, average = NUM_CLASSES, "multiclass", "macro"
        metric_collection = torchmetrics.MetricCollection({ 
            'Accuracy': torchmetrics.Accuracy(task=task, num_classes=num_classes, average=average, ignore_index=IGNORE_INDEX).to('cpu'),
            'Precision': torchmetrics.Precision(task=task, num_classes=num_classes, average=average, ignore_index=IGNORE_INDEX).to('cpu'), 
            'Recall': torchmetrics.Recall(task=task, num_classes=num_classes, average=average, ignore_index=IGNORE_INDEX).to('cpu'),
            "AUROC": torchmetrics.AUROC(task=task, num_classes=num_classes, average=average, ignore_index=IGNORE_INDEX).to('cpu'),
        }) 
        for i, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to('cpu')
            if enable_oneflow:
                # 不加to_global在过embedding时会报错，解决方案参考https://github.com/Oneflow-Inc/oneflow/pull/8894
                # model = model.to_local()  # 没有用，libai的vocab embedding强制要求global，虽然我也不知道为啥非得这么设计
                from libai.utils import distributed as dist
                input_ids = input_ids.to_global(placement=dist.get_layer_placement(0), sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]))
                attention_mask = flow.BoolTensor(attention_mask).to_global(placement=dist.get_layer_placement(0), sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]))

            no_grad_context = flow.no_grad if enable_oneflow else torch.no_grad
            with no_grad_context():
                preds = model(input_ids, attention_mask).permute(0, 2, 1)  # (N, C, T)
            if enable_oneflow:
                preds = torch.from_numpy(preds.numpy())#.to(DEVICE)
                labels = torch.from_numpy(labels.numpy())#.to(DEVICE)

            preds = preds.softmax(dim=1).cpu()
            batch_metrics = metric_collection.forward(preds, labels)
            if i % 20 == 0:
                for key, value in batch_metrics.items():
                    writer.write_log_single(f"eval/{key}(step)", value, steps)
        
        val_metrics = metric_collection.compute()
        print(val_metrics)
        for key, value in val_metrics.items():
            writer.write_log_single(f"eval/{key}", value, steps)
        metric_collection.reset()

    def train_model(model, train_data_loader, test_data_loader, loss_func, optimizer, model_name):
        dataset_size = len(train_data_loader.dataset)
        steps = 0
        for epoch in range(epochs):
            model.train()
            for i, batch in enumerate(train_data_loader):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                if enable_oneflow:
                    # 不加to_global在过embedding时会报错，解决方案参考https://github.com/Oneflow-Inc/oneflow/pull/8894
                    # model = model.to_local()  # 没有用，libai的vocab embedding强制要求global，虽然我也不知道为啥非得这么设计
                    from libai.utils import distributed as dist
                    input_ids = input_ids.to_global(placement=dist.get_layer_placement(0), sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]))
                    attention_mask = flow.BoolTensor(attention_mask).to_global(placement=dist.get_layer_placement(0), sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]))
                    labels = labels.to_global(placement=dist.get_layer_placement(0), sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]))
                
                preds = model(input_ids, attention_mask).permute(0, 2, 1)  # (N, C, T)
                loss = loss_func(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1
                if i % 20 == 0:
                    writer.write_log_single("train/loss", loss, steps)
                    writer.write_log_single("train/epoch", epoch, steps)
                    print(f'loss: {loss:>7f}  [epoch: {epoch} {i * BATCH_SIZE:>5d}/{dataset_size:>5d}]')

            evaluate(model, test_data_loader, steps, model_name)
        
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    train_model(model, train_data_loader, test_data_loader, loss_func, optimizer, model_name)