### adapted from https://github.com/pytorch/vision/tree/master/torchvision

import torch
import torch.nn as nn


#from .utils import load_state_dict_from_url
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.Sequential(
             nn.MaxPool2d(kernel_size=3, stride=2),
             nn.AdaptiveAvgPool2d((6, 6))
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        y  = self.avgpool(c5)
        y = torch.flatten(y, 1)
        f6 = self.fc6(y)
        f7 = self.fc7(f6)
        f8 = self.fc8(f7)
        return [c1, c2, c3, c4, c5, f6[:, :, None, None], f7[:, :, None, None], f8[:, :, None, None]]


def build_alexnet_fmaps(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        ### Rename dictionary keys to match new breakdown
        state_dict['conv1.0.weight'] = state_dict.pop('features.0.weight')
        state_dict['conv1.0.bias'] = state_dict.pop('features.0.bias')
        state_dict['conv2.0.weight'] = state_dict.pop('features.3.weight')
        state_dict['conv2.0.bias'] = state_dict.pop('features.3.bias')
        state_dict['conv3.1.weight'] = state_dict.pop('features.6.weight')
        state_dict['conv3.1.bias'] = state_dict.pop('features.6.bias')
        state_dict['conv4.0.weight'] = state_dict.pop('features.8.weight')
        state_dict['conv4.0.bias'] = state_dict.pop('features.8.bias')
        state_dict['conv5.0.weight'] = state_dict.pop('features.10.weight')
        state_dict['conv5.0.bias'] = state_dict.pop('features.10.bias')
        ###
        state_dict['fc6.0.weight'] = state_dict.pop('classifier.1.weight')
        state_dict['fc6.0.bias'] = state_dict.pop('classifier.1.bias')
        state_dict['fc7.0.weight'] = state_dict.pop('classifier.4.weight')
        state_dict['fc7.0.bias'] = state_dict.pop('classifier.4.bias')
        state_dict['fc8.0.weight'] = state_dict.pop('classifier.6.weight')
        state_dict['fc8.0.bias'] = state_dict.pop('classifier.6.bias')
        
        model.load_state_dict(state_dict)
    return model


class Alexnet_fmaps(nn.Module):
    '''
    image input dtype: float in range [0,1], size: 224, but flexible
    info on the dataloader compliant with the model database
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
    '''
    def __init__(self, pretrained=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Alexnet_fmaps, self).__init__()
        self.mean = nn.Parameter(torch.as_tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.as_tensor(std), requires_grad=False)
        self.extractor = build_alexnet_fmaps(pretrained=pretrained)

    def forward(self, _x):
        return self.extractor((_x - self.mean[None, :, None, None])/self.std[None, :, None, None])