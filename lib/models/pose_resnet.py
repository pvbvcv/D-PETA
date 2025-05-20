"""
Modified from https://github.com/microsoft/human-pose-estimation.pytorch
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn
from .resnet import _resnet
from .resnet import Bottleneck as Bottleneck_default
from torch.autograd import Variable
import torch
from .kp_class import *
from .DomainClassifier import *
from .vision_transformer import DinoVisionTransformer
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitl14 = torch.hub.load(repo_or_dir="/home/pengbaichao/.cache/torch/hub/facebookresearch_dinov2_main", model='dinov2_vitl14', source='local')
class Upsampling(nn.Sequential):
    """
    3-layers deconvolution used in `Simple Baseline <https://arxiv.org/abs/1804.06208>`_.
    """
    def __init__(self, in_channel=2048, hidden_dims=(256, 256), kernel_sizes=(4, 4), bias=False):
        assert len(hidden_dims) == len(kernel_sizes), \
            'ERROR: len(hidden_dims) is different len(kernel_sizes)'

        layers = []
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise NotImplementedError("kernel_size is {}".format(kernel_size))

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channel,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_channel = hidden_dim

        super(Upsampling, self).__init__(*layers)

        # init following Simple Baseline
        for name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class cwT(nn.Module):
    def __init__(self, in_channels, height, width):
        super(cwT, self).__init__()
        self.A = nn.Parameter(torch.zeros(in_channels))
        self.B = nn.Parameter(torch.zeros(in_channels))
        self.height = height
        self.width = width

    def forward(self, C):
        # 获取批次大小
        batch_size = C.size(0)
        
        # 将 A 和 B 扩展到与 C 匹配的形状
        A_expanded = self.A.view(1, -1, 1, 1)
        B_expanded = self.B.view(1, -1, 1, 1)
        
        # 按元素操作
        result = C * A_expanded + B_expanded
        result = result + C 
        return result

class PoseResNet(nn.Module):
    """
    `Simple Baseline <https://arxiv.org/abs/1804.06208>`_ for keypoint detection.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
    """
    def __init__(self, backbone, cwT, upsampling, feature_dim, num_keypoints, finetune=False):
        super(PoseResNet, self).__init__()
        self.backbone = backbone
        self.cwT = cwT
        self.upsampling = upsampling
        self.head = nn.Conv2d(in_channels=feature_dim, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0)
        self.finetune = finetune
        for m in self.head.modules():
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, return_fea=True):
        # if return_fea:
        #     x, cat_fea = self.backbone(x, return_fea)
            # print(cat_fea[0].size(), cat_fea[1].size(), cat_fea[2].size(), cat_fea[3].size())
            # exit(0)
        # else:
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        # x = self.cwT(x)
        # x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        x = self.cwT(x)
        x1 = self.upsampling(x)
        # print(x1.size())
        # exit(0)
        x = self.head(x1)
        if return_fea:
            return x, x1
        return x

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
        ]

def _pose_resnet(arch, num_keypoints, block, layers, pretrained_backbone, deconv_with_bias, finetune=False, progress=True, **kwargs):
    backbone = dinov2_vitl14
    # backbone = DinoVisionTransformer(
    #     img_size=518,
    #     patch_size=14,
    #     in_chans=3,
    #     embed_dim=1024,
    #     depth=24,
    #     num_heads=16,
    #     mlp_ratio=4.0,
    #     qkv_bias=True,
    #     ffn_bias=True,
    #     proj_bias=True,
    #     drop_path_rate=0.0,
    #     drop_path_uniform=False,
    #     init_values=1.0,  # for layerscale: None or 0 => no layerscale
    #     ffn_layer="mlp",
    #     block_chunks=0,
    #     num_register_tokens=0,
    #     interpolate_antialias=False,
    #     interpolate_offset=float(0.1),
    #                           )
    
    # checkpoint = torch.load("/home/pengbaichao/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth", map_location='cpu')
    # backbone.load_state_dict(checkpoint, strict=True)
    
    num_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"trainable parameters: {num_params}")
    
    CwT = cwT(backbone.embed_dim, 16, 16)
    upsampling = Upsampling(backbone.embed_dim, bias=deconv_with_bias)
    model = PoseResNet(backbone, CwT, upsampling, 256, num_keypoints, finetune)
    return model

def pose_resnet101(num_keypoints, pretrained_backbone=True, deconv_with_bias=False, finetune=False, progress=True, **kwargs):
    """Constructs a Simple Baseline model with a ResNet-101 backbone.

    Args:
        num_keypoints (int): number of keypoints
        pretrained_backbone (bool, optional): If True, returns a model pre-trained on ImageNet. Default: True.
        deconv_with_bias (bool, optional): Whether use bias in the deconvolution layer. Default: False
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True
    """
    return _pose_resnet('resnet101', num_keypoints, Bottleneck_default, [3, 4, 23, 3], pretrained_backbone, deconv_with_bias, finetune, progress, **kwargs)

def pose_resnet50(num_keypoints, pretrained_backbone=True, deconv_with_bias=False, finetune=False, progress=True, **kwargs):
    """Constructs a Simple Baseline model with a ResNet-101 backbone.

    Args:
        num_keypoints (int): number of keypoints
        pretrained_backbone (bool, optional): If True, returns a model pre-trained on ImageNet. Default: True.
        deconv_with_bias (bool, optional): Whether use bias in the deconvolution layer. Default: False
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True
    """
    return _pose_resnet('resnet50', num_keypoints, Bottleneck_default, [3, 4, 6, 3], pretrained_backbone, deconv_with_bias, finetune, progress, **kwargs)



def trans_pose_resnet(num_keypoints, arch=None, pretrained_backbone=True, deconv_with_bias=False, finetune=False, progress=True, **kwargs):
    """Constructs a Simple Baseline model with a ResNet-101 backbone.

    Args:
        num_keypoints (int): number of keypoints
        pretrained_backbone (bool, optional): If True, returns a model pre-trained on ImageNet. Default: True.
        deconv_with_bias (bool, optional): Whether use bias in the deconvolution layer. Default: False
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True
    """
    from .transpose_default import _C as cfg
    from .transpose_default import update_config

    update_config(cfg, kwargs['args'])
    model = get_pose_net_trans(cfg, arch, kwargs['args'].backbone, num_keypoints, args=kwargs['args'])
    model.finetune = finetune
    return model, cfg



def trans_pose_resnet50(num_keypoints, arch=None, pretrained_backbone=True, deconv_with_bias=False, finetune=False, progress=True, **kwargs):
    """Constructs a Simple Baseline model with a ResNet-101 backbone.

    Args:
        num_keypoints (int): number of keypoints
        pretrained_backbone (bool, optional): If True, returns a model pre-trained on ImageNet. Default: True.
        deconv_with_bias (bool, optional): Whether use bias in the deconvolution layer. Default: False
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True
    """
    from .transpose_default import _C as cfg
    from .transpose_default import update_config

    update_config(cfg, kwargs['args'])
    model = get_pose_net_trans50(cfg, arch)
    model.finetune = finetune
    return model, cfg

################################################################################################################

class PoseResNet_adv(nn.Module):
    """
    `Simple Baseline <https://arxiv.org/abs/1804.06208>`_ for keypoint detection.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
    """
    def __init__(self, backbone, upsampling, domain_classifier, feature_dim, num_keypoints, finetune=False):
        super(PoseResNet_adv, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = nn.Conv2d(in_channels=feature_dim, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0)
        self.domain_classifier = domain_classifier
        self.finetune = finetune
        for m in self.head.modules():
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, lambda_=None, return_fea=False):
        if return_fea:
            x, cat_fea = self.backbone(x, return_fea)
        else:
            x = self.backbone(x, return_fea)

        x1 = self.upsampling(x)
        x = self.head(x1)

        if lambda_ is not None:
            ReverseLayerF.apply(x1, lambda_)
            domain_output = self.domain_classifier(x1)
            return x, domain_output
        if return_fea:
            return x, cat_fea
        return x

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
        ]

def _pose_resnet_adv(arch, num_keypoints, block, layers, pretrained_backbone, deconv_with_bias, finetune=False, progress=True, **kwargs):
    backbone = _resnet(arch, block, layers, pretrained_backbone, progress, **kwargs)
    upsampling = Upsampling(backbone.out_features, bias=deconv_with_bias)
    domain_classifier = DomainClassifier()
    model = PoseResNet_adv(backbone, upsampling, domain_classifier, 256, num_keypoints, finetune)
    return model

def pose_resnet101_adv(num_keypoints, pretrained_backbone=True, deconv_with_bias=False, finetune=False, progress=True, **kwargs):
    """Constructs a Simple Baseline model with a ResNet-101 backbone.

    Args:
        num_keypoints (int): number of keypoints
        pretrained_backbone (bool, optional): If True, returns a model pre-trained on ImageNet. Default: True.
        deconv_with_bias (bool, optional): Whether use bias in the deconvolution layer. Default: False
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True
    """
    return _pose_resnet_adv('resnet101', num_keypoints, Bottleneck_default, [3, 4, 23, 3], pretrained_backbone, deconv_with_bias, finetune, progress, **kwargs)
