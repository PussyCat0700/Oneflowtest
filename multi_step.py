import oneflow as flow
import torch
import torchvision
import flowvision
from flowvision.datasets import CIFAR10
from utils import SeparateWriter

BATCH_SIZE = 64

def serious_train(writer:SeparateWriter, epochs:int, enable_oneflow:bool):
    DEVICE = 'cuda' if flow.cuda.is_available() else 'cpu'

    if enable_oneflow:
        import flowvision.transforms as transforms
        import oneflow.nn as nn
        from oneflow.optim import SGD
        from oneflow.utils.data import DataLoader
        writer.set_log_mode(SeparateWriter.ONEFLOW_ONLY)
    else:
        import torchvision.transforms as transforms
        import torch.nn as nn
        from torch.optim import SGD
        from torch.utils.data import DataLoader
        writer.set_log_mode(SeparateWriter.TORCH_ONLY)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CIFAR10(root='data', train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(root='data', train=False, transform=test_transform, download=True)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    if enable_oneflow:
        model = flowvision.models.resnet50(pretrained=True)
    else:
        model = torchvision.models.resnet50().to(DEVICE)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)

    def evaluate(model, data_loader, steps):
        dataset_size = len(data_loader.dataset)
        model.eval()
        num_corrects = 0
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images)
            if enable_oneflow:
                num_corrects += flow.sum(flow.argmax(preds, dim=1) == labels)
            else:
                num_corrects += torch.sum(torch.argmax(preds, dim=1) == labels)

        print('Accuracy: ', num_corrects.item() / dataset_size)
        writer.write_log_single("eval/Accuracy", num_corrects.item() / dataset_size, steps)

    def train_model(model, train_data_loader, test_data_loader, loss_func, optimizer):
        dataset_size = len(train_data_loader.dataset)
        steps = 0
        model.train()
        for epoch in range(epochs):
            for batch, (images, labels) in enumerate(train_data_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images)
                loss = loss_func(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1
                if batch % 100 == 0:
                    writer.write_log_single("train/loss", loss, steps)
                    writer.write_log_single("train/epoch", epoch, steps)
                    print(f'loss: {loss:>7f}  [epoch: {epoch} {batch * BATCH_SIZE:>5d}/{dataset_size:>5d}]')

            evaluate(model, test_data_loader, steps)
        
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss()
    train_model(model, train_data_loader, test_data_loader, loss_func, optimizer)