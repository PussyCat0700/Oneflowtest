# used to generate two same model.

import oneflow as flow
import flowvision
import torch
import torchvision
from config import cfgs

def generate_model(model_name="ResNet50"):
    if model_name not in cfgs['model_name']:
        print(f"Model {model_name} has not implement yet.")
        raise NotImplementedError()
    
    if model_name == 'ResNet50':
        tmodel = torchvision.models.resnet50()
        tmodel.fc = torch.nn.Linear(tmodel.fc.in_features, cfgs['ResNet50']['NUM_CLASSES'])
        fmodel = flowvision.models.resnet50()
        fmodel.fc = flow.nn.Linear(fmodel.fc.in_features, cfgs['ResNet50']['NUM_CLASSES'])
    elif model_name == "Inception":
        tmodel = torchvision.models.inception_v3()
        tmodel.fc = torch.nn.Linear(tmodel.fc.in_features, cfgs['Inception']['NUM_CLASSES'])
        fmodel = flowvision.models.inception_v3()
        fmodel.fc = flow.nn.Linear(fmodel.fc.in_features, cfgs['Inception']['NUM_CLASSES'])
    elif model_name == "SEResNet50":
        pass
    elif model_name == "MobileNet":
        pass
    elif model_name == "ShuffleNet":
        pass
    elif model_name == "DenseNet":
        pass
    elif model_name == "SwinTransformer":
        pass
    elif model_name == 'EfficientNet':
        pass

    return tmodel, fmodel
