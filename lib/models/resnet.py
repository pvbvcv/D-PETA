"""
Modified based on torchvision.models.resnet.
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""

import torch
import torch.nn as nn
from torchvision import models
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck
import copy
from torchvision.models.resnet import ResNet101_Weights
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


class ResNet(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features

    def forward(self, x, return_fea=False, return_up=False):
        """"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x) # 64
        x2 = self.layer2(x1)# 32
        x3 = self.layer3(x2)# 16
        x4 = self.layer4(x3)# 8
        if return_fea:
            x1 = self.layer1(x)  # 64
            x2 = self.layer2(x1) # 32
            x3 = self.layer3(x2) # 16
            x4 = self.layer4(x3) # 8

            if return_up:
                x2_up = nn.Upsample(scale_factor=2)(x2)
                x3_up = nn.Upsample(scale_factor=4)(x3)
                x4_up = nn.Upsample(scale_factor=8)(x4)
                fea_cat = torch.cat((x1, x2_up, x3_up, x4_up),0)
                # return x4, fea_cat
            else:

                fea_cat = [x1, x2, x3, x4]
            # print(x1.size(), x2.size(), x3.size(), x4.size())
            # print(x1.size(), x2_up.size(), x3_up.size(), x4_up.size())
            # print(fea_cat.size())
            # exit(0)
            return x4, fea_cat

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = x.view(-1, self._out_features)
        return x4

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):

    model = ResNet(block, layers, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(ResNet101_Weights.IMAGENET1K_V2.url,
                                              progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
