from utils.constants import *
import torch.nn as nn
import torchvision.models.vgg as vgg
import torchvision


def deeplabv3_segmentation(n_classes):
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=n_classes, aux_loss=None)
    model = model.to(DEVICE)

    return model

def resnet50_segmentation(n_classes):
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=n_classes, aux_loss=None)
    model = model.to(DEVICE)

    return model

def resnet101_segmentation(n_classes, pretrained=False):
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=pretrained, progress=True, num_classes=n_classes, aux_loss=None)
    model = model.to(DEVICE)

    return model

def resnet50(pretrained=False):
    model = torchvision.models.resnet50(pretrained=pretrained, progress=True)
    model = model.to(DEVICE)

    return model

def googlenet(pretrained=False):
    model = torchvision.models.googlenet(pretrained=pretrained, progress=True)
    model = model.to(DEVICE)

    return model

def VGG_19():
    VGG = vgg.vgg19(pretrained=True)
    VGG = VGG.to(DEVICE)

    return VGG.eval()

def inceptionv3(pretrained = False):
    model = torchvision.models.inception_v3(pretrained=pretrained, progress=True)
    model = model.to(DEVICE)

    return model


if __name__ == '__main__':

    model = googlenet(pretrained=False)
    model.conv1.conv = nn.Conv2d(17, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(DEVICE)
    print(model)
