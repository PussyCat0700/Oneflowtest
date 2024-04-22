# used to generate two same model.

import oneflow as flow
import flowvision
import torch
import torchvision
from config import cfgs

from pymodels.seresnet50 import se_resnet50
from pymodels.swin_tiny import swin_tiny_patch4_window7_224

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
        tmodel = se_resnet50()
        tmodel.last_linear = torch.nn.Linear(2048, cfgs['SEResNet50']['NUM_CLASSES'])
        fmodel = flowvision.models.se_resnet50()
        fmodel.last_linear = flow.nn.Linear(2048, cfgs['SEResNet50']['NUM_CLASSES'])

    elif model_name == "MobileNet":
        tmodel = torchvision.models.mobilenet_v2()
        tmodel.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(1280, cfgs['MobileNet']['NUM_CLASSES']))
        fmodel = flowvision.models.mobilenet_v2()
        fmodel.classifier = flow.nn.Sequential(flow.nn.Dropout(0.2), flow.nn.Linear(1280, cfgs['MobileNet']['NUM_CLASSES']))

    elif model_name == "ShuffleNet":
        tmodel = torchvision.models.shufflenet_v2_x0_5()
        tmodel.fc = torch.nn.Linear(1024, cfgs['ShuffleNet']['NUM_CLASSES'])
        fmodel = flowvision.models.shufflenet_v2_x0_5()
        fmodel.fc = flow.nn.Linear(1024, cfgs['ShuffleNet']['NUM_CLASSES'])
        
    elif model_name == "DenseNet":
        tmodel = torchvision.models.densenet121()
        tmodel.classifier = torch.nn.Linear(1024, cfgs['DenseNet']['NUM_CLASSES'])
        fmodel = flowvision.models.densenet121()
        fmodel.classifier = flow.nn.Linear(1024, cfgs['DenseNet']['NUM_CLASSES'])

    elif model_name == "SwinTransformer":
        tmodel = swin_tiny_patch4_window7_224()
        tmodel.head = torch.nn.Linear(768, cfgs['SwinTransformer']['NUM_CLASSES'])
        fmodel = flowvision.models.swin_tiny_patch4_window7_224()
        fmodel.head = flow.nn.Linear(768, cfgs['SwinTransformer']['NUM_CLASSES'])

    elif model_name == 'EfficientNet':
        tmodel = torchvision.models.efficientnet_b2()
        tmodel.classifier = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(1408, cfgs['EfficientNet']['NUM_CLASSES']))
        fmodel = flowvision.models.efficientnet_b2()
        fmodel.classifier = flow.nn.Sequential(flow.nn.Dropout(0.3), flow.nn.Linear(1408, cfgs['EfficientNet']['NUM_CLASSES']))

    state_dict = {k: v.cpu().numpy() for k, v in tmodel.state_dict().items()}
    fmodel.load_state_dict({k: flow.tensor(v) for k, v in state_dict.items()})
    return tmodel, fmodel
