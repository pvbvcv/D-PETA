# help functions are adapted from original mean teacher network
# https://github.com/CuriousAI/mean-teacher/tree/master/pytorch

from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.optim import Optimizer
import torch.nn as nn
from skimage.filters import threshold_otsu
import argparse
import torchvision.transforms.functional as tF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import kl_div

class OldWeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, target_net, source_net, alpha=0.999):
        self.target_params = list(target_net.parameters())
        self.source_params = list(source_net.parameters())
        self.alpha = alpha
        self.target_net = target_net
        self.source_net = source_net
        for p, src_p in zip(self.target_params, self.source_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.target_params, self.source_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
        self.target_net.module.cwT.load_state_dict(self.source_net.module.cwT.state_dict())


class WeightEMASelectiveUpdate(object):
    """
    Exponential moving average weight optimizer for mean teacher model,
    selectively updating upsampling and head parts.
    """
    def __init__(self, target_net, source_net, alpha=0.999):
        self.alpha = alpha

        # Extract parameters from upsampling and head parts
        self.target_upsampling_params = list(target_net.module.upsampling.parameters())
        self.source_upsampling_params = list(source_net.module.upsampling.parameters())
        

        self.target_head_params = list(target_net.module.head.parameters())
        self.source_head_params = list(source_net.module.head.parameters())


        # Initialize target parameters to match source parameters
        for p, src_p in zip(self.target_upsampling_params, self.source_upsampling_params):
            p.data[:] = src_p.data[:]
        
        for p, src_p in zip(self.target_head_params, self.source_head_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        
        # Update upsampling parameters
        for p, src_p in zip(self.target_upsampling_params, self.source_upsampling_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
        
        # Update head parameters
        for p, src_p in zip(self.target_head_params, self.source_head_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)



class MultiWeightEMA(object):
    """
    Exponential moving average weight optimizer for multiple mean teacher models
    """
    def __init__(self, target_nets, source_net, alpha=0.999):
        """
        :param target_nets: List of target networks (teacher models)
        :param source_net: Source network (student model)
        :param alpha: Exponential moving average coefficient
        """
        self.target_params_list = [list(target_net.parameters()) for target_net in target_nets]
        self.source_params = list(source_net.parameters())
        self.alpha = alpha

        # 初始化时同步每个目标网络和源网络的参数
        for target_params in self.target_params_list:
            for p, src_p in zip(target_params, self.source_params):
                p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for target_params in self.target_params_list:
            for p, src_p in zip(target_params, self.source_params):
                p.data.mul_(self.alpha)
                p.data.add_(src_p.data * one_minus_alpha)



class CombinedWeightEMA(object):
    """
    Exponential moving average weight optimizer using two student models to update one teacher model
    """
    def __init__(self, teacher_net, student_net1, student_net2, alpha=0.999):
        self.teacher_params = list(teacher_net.parameters())
        self.student_params1 = list(student_net1.parameters())
        self.student_params2 = list(student_net2.parameters())
        self.alpha = alpha / 2  # 每个学生模型的权重为原先的一半

        # 初始化时同步教师模型和第一个学生模型的参数
        for p, src_p1 in zip(self.teacher_params, self.student_params1):
            p.data[:] = src_p1.data[:]

    def step(self):
        one_minus_alpha = 1.0 - 2 * self.alpha  # 调整后的剩余权重
        for p, src_p1, src_p2 in zip(self.teacher_params, self.student_params1, self.student_params2):
            p.data.mul_(self.alpha * 2)  # 保持现有参数的部分权重
            p.data.add_(src_p1.data * self.alpha)  # 加入第一个学生模型的参数
            p.data.add_(src_p2.data * self.alpha)  # 加入第二个学生模型的参数
            
class DualWeightEMA(object):
    """
    Exponential moving average weight optimizer for dual mean teacher model
    """
    def __init__(self, target_net1, target_net2, source_net, alpha=0.999):
        self.target_params1 = list(target_net1.parameters())
        self.target_params2 = list(target_net2.parameters())
        self.source_params = list(source_net.parameters())
        self.alpha = alpha

        # 初始化时同步目标网络1和源网络的参数
        for p, src_p in zip(self.target_params1, self.source_params):
            p.data[:] = src_p.data[:]
        
        # 初始化时同步目标网络2和源网络的参数
        for p, src_p in zip(self.target_params2, self.source_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p1, p2, src_p in zip(self.target_params1, self.target_params2, self.source_params):
            p1.data.mul_(self.alpha)
            p1.data.add_(src_p.data * one_minus_alpha)
            
            p2.data.mul_(self.alpha)
            p2.data.add_(src_p.data * one_minus_alpha)
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    # assert 0 <= current <= rampdown_length
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def rev_sigmoid(progress):

    progress = np.clip(progress, 0, 1)
    return float(1. / (1 + np.exp(10 * progress - 5)))

def sigmoid(progress):

    progress = np.clip(progress, 0, 1)
    return float(1. / (1 + np.exp(5 - 10 * progress)))

def get_max_preds_torch(batch_heatmaps):

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    width = batch_heatmaps.size(3)
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(heatmaps_reshaped, 2)
    maxvals = torch.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = (maxvals > 0.0).repeat(1, 1, 2)
    pred_mask = pred_mask.float()

    preds *= pred_mask
    return preds, maxvals

def rectify(hm, sigma): # b, c, h, w -> b, c, h, w
    b, c, h, w = hm.size()
    rec_hm = torch.zeros_like(hm)
    pred_coord, pred_val = get_max_preds_torch(hm) # b, c, 2
    tmp_size = 3 * sigma
    for b in range(rec_hm.size(0)):
        for c in range(rec_hm.size(1)):
            mu_x = pred_coord[b, c, 0]
            mu_y = pred_coord[b, c, 1]
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if mu_x >= h or mu_y >= w or mu_x < 0 or mu_y < 0:
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = torch.arange(0, size, 1).float()
            y = x.unsqueeze(1)
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], h)
            img_y = max(0, ul[1]), min(br[1], w)

            rec_hm[b][c][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return rec_hm

def generate_prior_map(prior, preds, gamma=2, sigma=2, epsilon=-10e10, v3=False): # prior: {mean: (k, k), std: (k, k)}, preds: (b, k, h, w) -> returns prior_map: (b, k, h, w)
    # for the prediction in each channel, generate the estimation of the rest channels (assign a weight for each according to confidence and std?) with shape of (k, k, h, w)
    # ensemble all the estimation and form a prior map, which should be a multiplier for the original prediction map.

    prior_mean = prior['mean'].cuda()
    prior_std = prior['std'].cuda()
    B, K, H, W = preds.size()
    pred_coord, pred_val = get_max_preds_torch(preds) # B, K, (1), 2 ; B, K, 1
    pred_coord = pred_coord.view(B,K,1,2,1,1)

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,1,H,W).repeat(B,K,1,1,1)
    yy = yy.view(1,1,1,H,W).repeat(B,K,1,1,1)
    grid = torch.cat((xx,yy),2).float().cuda().view(B,1,K,2,H,W) # B, (1), K, 2, H, W

    dist = torch.norm(grid - pred_coord, dim=3) # B, K, K, H, W
    dist -= prior_mean.view(1,K,K,1,1) # B, K, K, H, W
    targets = torch.exp(-(dist**2) / (2 * sigma ** 2)) # B, K, K, H, W

    if v3:
        var_table = (1 / (1 + prior_std)).view(1,K,K) # 1, K, K
        conf_table = pred_val.view(B,K,1) # B, K, 1
        final_weight = var_table * conf_table # B, K, K
        # final_weight = F.softmax(final_weight, dim=1) # B, K, K, 1
        targets = torch.sum(final_weight.view(B, K, K, 1, 1) * targets, dim=1)

    else:
        temp_std = -prior_std / gamma
        temp_std.fill_diagonal_(epsilon)
        weights = F.softmax(temp_std, dim=0) # K, K

        targets = torch.sum(weights.view(1, K, K, 1, 1) * targets, dim=1)

    return targets


def getmark(y_t_stu_recon, y_t_tea_recon):
    criterion = nn.CrossEntropyLoss()
    saveloss = torch.zeros(y_t_stu_recon.shape[0])
    # sign = torch.zeros(y_t_stu_recon[0])
    for i in range(y_t_stu_recon.shape[0]):
        saveloss[i] = criterion(y_t_stu_recon[i], y_t_tea_recon[i])
    threshold = threshold_otsu(saveloss.cpu().detach().numpy())
    marks = (saveloss.cpu().detach().numpy() <= threshold).astype(int)
    return marks


def decorrelation_loss(features):
    # 确保 features 是 4D 张量
    assert features.dim() == 4, "Features should be a 4D tensor"

    # 将特征展平到二维矩阵形式 (batch_size * height * width, channels)
    batch_size, channels, height, width = features.size()
    features = features.permute(0, 2, 3, 1).contiguous().view(-1, channels)

    # 去均值
    features = features - features.mean(dim=0, keepdim=True)

    # 计算协方差矩阵
    cov = torch.mm(features.T, features) / features.size(0)

    # 创建单位矩阵
    I = torch.eye(cov.size(0)).to(features.device)

    # Frobenius 范数损失
    return torch.norm(cov - I)


def kl_divergence(pred1, pred2):
    temp1 = pred1.clone().cpu().numpy()
    temp2 = pred2.clone().cpu().numpy()
    return np.sum(kl_div(temp1, temp2))


def cosine_similarity_score(pred1, pred2):
    temp1 = pred1.clone().cpu()
    temp2 = pred2.clone().cpu()
    return cosine_similarity(temp1.reshape(1, -1), temp2.reshape(1, -1))[0][0].item()

def getPI(tensor1, tensor2):
    def unravel_index(indices, shape):
        rows = indices // shape[1]
        cols = indices % shape[1]
        return rows, cols

    # 找到每个batch和每个通道的最大值索引
    tensor11 = torch.tensor(tensor1.clone().detach())
    tensor22 = torch.tensor(tensor2.clone().detach())
    max_indices1 = torch.argmax(tensor11.view(16, 16, -1), dim=2)
    max_indices2 = torch.argmax(tensor22.view(16, 16, -1), dim=2)

    # 将扁平化的索引转换为二维索引
    max_indices_2d1 = torch.stack([torch.tensor(unravel_index(idx, (64, 64))) for i in range(16) for j in range(16) for idx in [max_indices1[i, j]]])
    max_indices_2d2 = torch.stack([torch.tensor(unravel_index(idx, (64, 64))) for i in range(16) for j in range(16) for idx in [max_indices2[i, j]]])

    # 重新调整形状
    max_indices_2d1 = max_indices_2d1.view(16, 16, 2)
    max_indices_2d2 = max_indices_2d2.view(16, 16, 2)

    # 计算两个索引组的差值的绝对值
    index_diff_abs = torch.abs(max_indices_2d1 - max_indices_2d2)

    # 对差值的绝对值求和
    sum_diff_abs = index_diff_abs.sum()

    # 计算热图对角线长度
    heatmap_diagonal_length = (64**2 + 64**2) ** 0.5

    # 计算最终结果
    result = sum_diff_abs / heatmap_diagonal_length

    return result.item()

def getmark(y_t_stu_recon, y_t_tea_recon):
    criterion = nn.CrossEntropyLoss()
    saveloss = torch.zeros(y_t_stu_recon.shape[0])
    # sign = torch.zeros(y_t_stu_recon[0])
    for i in range(y_t_stu_recon.shape[0]):
        saveloss[i] = criterion(y_t_stu_recon[i], y_t_tea_recon[i])
    threshold = threshold_otsu(saveloss.cpu().detach().numpy())
    marks = (saveloss.cpu().detach().numpy() <= threshold).astype(int)
    return marks


def random_mask_tensor_images(images, mask_size=(50, 50)):
    """
    对一批张量形式的图片进行随机遮盖
    :param images: 张量形式的图片，形状为 (batch_size, channels, height, width)
    :param mask_size: 遮盖区域的大小 (width, height)
    :return: 遮盖后的图片张量和遮盖区域的坐标
    """
    channels, height, width = images.shape
    masked_images = images.clone()  # 复制原始张量
    mask_regions = []  # 记录每张图片的遮盖区域


    # 随机生成遮盖区域左上角 (x, y)
    x_start = random.randint(0, max(0, width - mask_size[0]))
    y_start = random.randint(0, max(0, height - mask_size[1]))
    
    # 确定遮盖区域右下角
    x_end = x_start + mask_size[0]
    y_end = y_start + mask_size[1]
    
    # 对图片张量进行遮盖（设置为 0 表示黑色）
    masked_images[:, y_start:y_end, x_start:x_end] = 0
    mask_regions.append((x_start, y_start, x_end, y_end))
    
    return masked_images


def splice_images_random(img1, img2, split_ratio_range=(0.3, 0.7)):
    """
    随机选择水平或垂直方向，按指定比例拼接图片。
    :param images: 一个包含图片张量的列表，每个图片形状为 (C, H, W)
    :param split_ratio: 拼接比例，默认按 50:50（即 0.5）
    :return: 拼接后的新图片张量
    """
    # 检查所有图片形状是否一致
    if img1.shape != img2.shape:
        raise ValueError("所有图片的形状必须一致！")
    
    # 获取图片形状
    C, H, W = img1.shape
    
    # 随机选择拼接方向（50% 概率水平拼接或垂直拼接）
    direction = 'vertical' if random.random() < 0.5 else 'horizontal'
    split_dim = 1 if direction == 'vertical' else 2  # 选择拼接维度（1=高度，2=宽度）

    # 随机选择拼接比例
    split_ratio = random.uniform(*split_ratio_range)

    # 分割图片
    split_parts = []
    if direction == 'vertical':
        split_index = int(H * split_ratio)
        split_parts.append(img1[:, :split_index, :])  # 上半部分
        split_parts.append(img2[:, split_index:, :])  # 下半部分
    elif direction == 'horizontal':
        split_index = int(W * split_ratio)
        split_parts.append(img1[:, :, :split_index])  # 左半部分
        split_parts.append(img2[:, :, split_index:])  # 右半部分

    # 按照方向进行拼接
    combined_image = torch.cat(split_parts, dim=split_dim)
    
    return combined_image


def get_simple_hard_samples(student, teacher, iter, args: argparse.Namespace):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t_ori_stu, x_t_ori_tea, x_t_stu, _, _, meta_t_stu, x_t_teas, _, _, meta_t_tea = next(iter)
    x_t_ori_stu = x_t_ori_stu.to(device)
    x_t_ori_tea = x_t_ori_tea.to(device)
    x_t_stu = x_t_stu.to(device)
    x_t_teas = [x_t_tea.to(device) for x_t_tea in x_t_teas]
    ratio = args.image_size / args.heatmap_size
    saveloss = []
    for i in tqdm(range(125)):
        with torch.no_grad():
            # -----------------get_augment1_simple--------------
            y_t_teas = [student(x_t_tea) for x_t_tea in x_t_teas] # softmax on w, h
            y_t_tea_recon = torch.zeros_like(y_t_teas[0][0]).cuda() # b, c, h, w
            tea_mask = torch.zeros(y_t_teas[0][0].shape[:2]).cuda() # b, c
            for ind in range(x_t_teas[0].size(0)):
                recons = torch.zeros(args.k, *y_t_teas[0][0].size()[1:]) # k, c, h, w
                for _k in range(args.k):
                    angle, [trans_x, trans_y], [shear_x, shear_y], scale = meta_t_tea[_k]['aug_param_tea']
                    _angle, _trans_x, _trans_y, _shear_x, _shear_y, _scale = angle[ind].item(), trans_x[ind].item(), trans_y[ind].item(), shear_x[ind].item(), shear_y[ind].item(), scale[ind].item() 
                    temp = tF.affine(y_t_teas[0][_k][ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                    temp = tF.affine(temp, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                    temp = tF.affine(temp, 0., translate=[0, 0], shear=[_shear_x, _shear_y], scale=1.) # c, h, w
                    recons[_k] = temp # c, h, w

                y_t_tea_recon[ind] = torch.mean(recons, dim=0) # (c, h, w)
            y_t_tea_recon = rectify(y_t_tea_recon, sigma=args.sigma)

            # -----------------get_augment2_simple--------------
            angle, [trans_x, trans_y], [shear_x, shear_y], scale = meta_t_stu['aug_param_stu']
            y_t_stu, y_t_stu_fea = student(x_t_stu) # softmax on w, h          
            y_t_stu_recon = torch.zeros_like(y_t_stu).cuda() # b, c, h, w
            for ind in range(x_t_stu.size(0)):
                _angle, _trans_x, _trans_y, _shear_x, _shear_y, _scale = angle[ind].item(), trans_x[ind].item(), trans_y[ind].item(), shear_x[ind].item(), shear_y[ind].item(), scale[ind].item()
                temp = tF.affine(y_t_stu[ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                temp = tF.affine(temp, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                y_t_stu_recon[ind] = tF.affine(temp, 0., translate=[0., 0.], shear=[_shear_x, _shear_y], scale=1.)

            # -----------------get_augment3_simple--------------
            x_t_ori_stu_occlude = torch.zeros_like(x_t_stu)
            for i in range(args.batch_size):
                x_t_ori_stu_occlude[i] = random_mask_tensor_images(x_t_ori_stu[i], mask_size=(40, 40))
            y3, _ = student(x_t_ori_stu_occlude)
            
            # -----------------get_augment4_simple--------------
            y4, _ = student(x_t_ori_tea)



            # -----------------get_augment1_simple--------------
            y_t_teas1 = [teacher(x_t_tea) for x_t_tea in x_t_teas] # softmax on w, h
            y_t_tea_recon1 = torch.zeros_like(y_t_teas1[0][0]).cuda() # b, c, h, w
            tea_mask = torch.zeros(y_t_teas1[0][0].shape[:2]).cuda() # b, c
            for ind in range(x_t_teas[0].size(0)):
                recons = torch.zeros(args.k, *y_t_teas1[0][0].size()[1:]) # k, c, h, w
                for _k in range(args.k):
                    angle, [trans_x, trans_y], [shear_x, shear_y], scale = meta_t_tea[_k]['aug_param_tea']
                    _angle, _trans_x, _trans_y, _shear_x, _shear_y, _scale = angle[ind].item(), trans_x[ind].item(), trans_y[ind].item(), shear_x[ind].item(), shear_y[ind].item(), scale[ind].item() 
                    temp = tF.affine(y_t_teas1[0][_k][ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                    temp = tF.affine(temp, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                    temp = tF.affine(temp, 0., translate=[0, 0], shear=[_shear_x, _shear_y], scale=1.) # c, h, w
                    recons[_k] = temp # c, h, w

                y_t_tea_recon1[ind] = torch.mean(recons, dim=0) # (c, h, w)
            y_t_tea_recon1 = rectify(y_t_tea_recon1, sigma=args.sigma)

            # -----------------get_augment2_simple--------------
            angle, [trans_x, trans_y], [shear_x, shear_y], scale = meta_t_stu['aug_param_stu']
            y_t_stu1, y_t_stu_fea = teacher(x_t_stu) # softmax on w, h          
            y_t_stu_recon1 = torch.zeros_like(y_t_stu1).cuda() # b, c, h, w
            for ind in range(x_t_stu.size(0)):
                _angle, _trans_x, _trans_y, _shear_x, _shear_y, _scale = angle[ind].item(), trans_x[ind].item(), trans_y[ind].item(), shear_x[ind].item(), shear_y[ind].item(), scale[ind].item()
                temp = tF.affine(y_t_stu1[ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                temp = tF.affine(temp, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                y_t_stu_recon1[ind] = tF.affine(temp, 0., translate=[0., 0.], shear=[_shear_x, _shear_y], scale=1.)

            # -----------------get_augment3_simple--------------
            x_t_ori_stu_occlude = torch.zeros_like(x_t_stu)
            for i in range(args.batch_size):
                x_t_ori_stu_occlude[i] = random_mask_tensor_images(x_t_ori_stu[i], mask_size=(40, 40))
            y31, _ = teacher(x_t_ori_stu_occlude)
            
            # -----------------get_augment4_simple--------------
            y41, _ = teacher(x_t_ori_tea)

            for i in range(y4.shape[0]):
                loss = (criterion(y_t_stu_recon[i], y3[i]) + criterion(y_t_stu_recon[i], y4[i]) + criterion(y3[i], y4[i]) + criterion(y_t_tea_recon[i], y_t_stu_recon[i]) + criterion(y_t_tea_recon[i], y3[i]) + criterion(y_t_tea_recon[i], y4[i]))/6
                loss1 = (criterion(y_t_stu_recon1[i], y31[i]) + criterion(y_t_stu_recon1[i], y41[i]) + criterion(y31[i], y41[i]) + criterion(y_t_tea_recon1[i], y_t_stu_recon1[i]) + criterion(y_t_tea_recon1[i], y31[i]) + criterion(y_t_tea_recon1[i], y41[i]))/6  
                loss2 = loss + loss1
                saveloss.append(loss2)
    saveloss = torch.tensor(saveloss)
    threshold = threshold_otsu(saveloss.cpu().detach().numpy())
    marks = (saveloss.cpu().detach().numpy() <= threshold).astype(int)
    return marks