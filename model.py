import torch
from torch import nn
import torchvision


def create_model(model_name="resnet34", num_classes=3):
    """
    :param str model_name: The name of the model, for instance 'resnetXX'
    :param int num_classes: The number of classes to predict. This will
        replace the last Linear layer of the pretrained models
    :return: pretrained pytorch vision model
    """
    model = None
    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == "resnet34":
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == "vit_b_16":
        model = torchvision.models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(768, num_classes)
    elif model_name == "convnext_tiny":
        model = torchvision.models.convnext_tiny(pretrained=True)
        model.classifier[2] = nn.Linear(768, num_classes)
    elif model_name == "efficientnet_b3":
        model = torchvision.models.efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(1536, num_classes)
    elif model_name == "efficientnet_b5":
        model = torchvision.models.efficientnet_b5(pretrained=True)
        model.classifier[1] = nn.Linear(2048, num_classes)
    assert model is not None
    return model
