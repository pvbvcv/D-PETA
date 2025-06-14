from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet101_Weights
from torchvision.models.resnet import BasicBlock, Bottleneck
from .functions import ReverseLayerF


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
    34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
    50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
    101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
    152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')
}

class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, in_channel=3):
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class DeconvHead(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters, kernel_size, conv_kernel_size, num_joints, depth_dim,
                 with_bias_end=True):
        super(DeconvHead, self).__init__()

        conv_num_filters = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert conv_kernel_size == 1 or conv_kernel_size == 3, 'Only support kenerl 1 and 3'
        if conv_kernel_size == 1:
            pad = 0
        elif conv_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                _in_channels = in_channels
                self.features.append(
                    nn.Conv2d(_in_channels, num_filters, kernel_size=1, stride=1, bias=False))
                self.features.append(nn.BatchNorm2d(num_filters))
                self.features.append(nn.ReLU(inplace=True))
            else:
                _in_channels = num_filters
                self.features.append(
                    nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                       output_padding=output_padding, bias=False))
                self.features.append(nn.BatchNorm2d(num_filters))
                self.features.append(nn.ReLU(inplace=True))

        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(conv_num_filters))
            self.features.append(nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        features = []
        for i, l in enumerate(self.features):
            x = l(x)
            if (i+1) % 3 == 0:
                features.append(x)
        return x, features


class DomainClassifier(nn.Module):
    def __init__(self, input_dim=1024, ndf=64, with_bias=False):
        super(DomainClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, ndf, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv6 = nn.Conv2d(ndf * 16, 1, kernel_size=2, stride=1, bias=with_bias)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.leaky_relu(x)
        x = self.conv6(x)
        return x


class Bottleneck_refinenet(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck_refinenet, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 2,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 2),
        )

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class refineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class, dual_branch=False):
        super(refineNet, self).__init__()
        cascade = []
        num_cascade = 4
        for i in range(num_cascade):
            cascade.append(self._make_layer(lateral_channel, num_cascade-i-1, out_shape))
        self.cascade = nn.ModuleList(cascade)
        self.final_predict = self._predict(4 * lateral_channel, num_class)
        self.dual_branch = dual_branch
        if dual_branch:
            self.final_predict_2 = self._predict(4 * lateral_channel, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def _make_layer(self, input_channel, num, output_shape):
        layers = []
        for i in range(num):
            layers.append(Bottleneck_refinenet(input_channel, 128))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, input_channel, num_class, with_bias_end=True):
        layers = []
        layers.append(Bottleneck_refinenet(input_channel, 128))
        layers.append(nn.Conv2d(256, num_class, kernel_size=3, stride=1, padding=1, bias=with_bias_end))
        return nn.Sequential(*layers)

    def forward(self, x):
        refine_fms = []
        for i in range(4):
            refine_fms.append(self.cascade[i](x[i]))
        out = torch.cat(refine_fms, dim=1)
        out1 = self.final_predict(out)
        if self.dual_branch:
            out2 = self.final_predict_2(out)
            return [out1, out2], out
        else:
            return out1, out


class ResPoseNet_refine_mt(nn.Module):
    def __init__(self, backbone, head, domain_classifier, refinenet):
        super(ResPoseNet_refine_mt, self).__init__()
        self.backbone = backbone
        self.head = head
        self.domain_classifier = domain_classifier
        self.refinenet = refinenet

    def forward(self, x, lambda_=None, return_domain=True):
        # lambda_ is used to
        x = self.backbone(x)
        x, features = self.head(x)
        x_refine, ft = self.refinenet(features)
        if return_domain:
            ft_rev = ReverseLayerF.apply(ft, lambda_)
            domain_output = self.domain_classifier(ft_rev)
            return x, x_refine, domain_output
        else:
            return x, x_refine


def get_default_network_config():
    config = edict()
    config.from_model_zoo = True
    config.pretrained = ''
    config.num_layers = 101
    config.num_deconv_layers = 4
    config.num_deconv_filters = 256
    config.num_deconv_kernel = 4
    config.final_conv_kernel = 1
    config.depth_dim = 1
    config.input_channel = 3
    return config


def init_pose_net(pose_net, name):
    org_resnet = model_zoo.load_url(ResNet101_Weights.IMAGENET1K_V2.url)
    # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
    org_resnet.pop('fc.weight', None)
    org_resnet.pop('fc.bias', None)
    pose_net.backbone.load_state_dict(org_resnet)
    print("Init Network from model zoo")


def init_pretrained(pose_net, pretrained_path):
    checkpoint = torch.load(pretrained_path)
    print("=> loading pretrained checkpoint '{}' (epoch {})".format(pretrained_path, checkpoint['epoch']))
    pretrained_dict = checkpoint['state_dict']
    model_dict = pose_net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    params_refine_pred = [p for p in pretrained_dict if p.startswith('module.refinenet.final_predict')]
    for p in params_refine_pred:
        p_model = p.replace('final_predict', 'final_predict_2')
        model_dict[p_model] = pretrained_dict[p]  # initialize the both heads with the pretrained one head
    pose_net.load_state_dict(model_dict)
    print("Init Network from pretrained pose_resnet_refinenet")


def pose_resnet_refine_mt_multida(**kwargs):

    cfg = get_default_network_config()

    block_type, layers, channels, name = resnet_spec[kwargs['resnet_layers']]
    backbone_net = ResNetBackbone(block_type, layers)
    head_net = DeconvHead(
        channels[-1], cfg.num_deconv_layers,
        cfg.num_deconv_filters, cfg.num_deconv_kernel,
        cfg.final_conv_kernel, kwargs['num_classes'], cfg.depth_dim
    )
    domain_classifier = DomainClassifier()
    refinenet = refineNet(256, (64, 64), kwargs['num_classes'], dual_branch=kwargs['dual_branch'])
    pose_net = ResPoseNet_refine_mt(backbone_net, head_net, domain_classifier, refinenet)

    return pose_net

