import random
import time
import warnings
import sys
import argparse
import shutil
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import shutil
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage
import torch.nn.functional as F
import torchvision.transforms.functional as tF
import lib.models as models
from lib.models.loss import JointsMSELoss, ConsLoss, EntLoss, DivLoss
import lib.datasets as datasets
import lib.transforms.keypoint_detection as T
from lib.transforms import Denormalize
from lib.data import ForeverDataIterator
from lib.meter import AverageMeter, ProgressMeter, AverageMeterDict, AverageMeterList
from lib.keypoint_detection import accuracy
from lib.logger import CompleteLogger
from lib.models import Style_net
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recover_min = torch.tensor([-2.1179, -2.0357, -1.8044]).to(device)
recover_max = torch.tensor([2.2489, 2.4285, 2.64]).to(device)

def P_shuffle(image_tensors):
    """
    随机打乱每张图片中4x4大小的patches。
    输入是形状为(batch_size, channels, height, width)的图片张量。
    """
    batch_size, channels, height, width = image_tensors.shape
    patch_size = 16
    # 分割图片为4x4的patches
    # 使用unfold方法分割每张图片
    patches = image_tensors.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)  # 重整形状

    # 随机打乱patches
    # 对每张图片的patches进行随机打乱
    shuffled_images = []
    for i in range(batch_size):
        num_patches = patches.shape[2]
        indices = torch.randperm(num_patches)
        shuffled_patches = patches[i, :, indices, :, :]
        # 重组图片
        rows = cols = height // patch_size
        shuffled_image = shuffled_patches.view(channels, rows, cols, patch_size, patch_size)
        shuffled_image = shuffled_image.permute(0, 1, 3, 2, 4).contiguous()
        shuffled_image = shuffled_image.view(channels, height, width)
        shuffled_images.append(shuffled_image.unsqueeze(0))
    
    # 将列表中的所有图片张量合并成一个批次
    return torch.cat(shuffled_images, dim=0)

def img_trans():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    src_train_transform = T.Compose([
        T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
        T.RandomAffineRotation(args.rotation_stu, args.shear_stu, args.translate_stu, args.scale_stu),
        T.ColorJitter(brightness=args.color_stu, contrast=args.color_stu, saturation=args.color_stu),
        T.GaussianBlur(high=args.blur_stu),
        T.ToTensor(),
        normalize
    ])
   
    base_transform = T.Compose([
        T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
    ])
    label_transform = T.Compose([
        T.ToTensor(),
        normalize
    ])
    rotate_transform = T.Compose([
        T.RandomAffineRotation(args.rotation_stu, args.shear_stu, args.translate_stu, args.scale_stu)
    ])
    tgt_train_transform_ori_stu = T.Compose([
        # T.RandomAffineRotation(args.rotation_stu, args.shear_stu, args.translate_stu, args.scale_stu),
        T.ColorJitter(brightness=args.color_stu, contrast=args.color_stu, saturation=args.color_stu),
        T.GaussianBlur(high=args.blur_stu),
        T.ToTensor(),
        normalize
    ])
    tgt_train_transform_ori_tea = T.Compose([
        # T.RandomAffineRotation(args.rotation_tea, args.shear_tea, args.translate_tea, args.scale_tea),
        T.ColorJitter(brightness=args.color_tea, contrast=args.color_tea, saturation=args.color_tea),
        T.GaussianBlur(high=args.blur_tea),
        T.ToTensor(),
        normalize
        ])
    tgt_train_transform_stu = T.Compose([
        T.RandomAffineRotation(args.rotation_stu, args.shear_stu, args.translate_stu, args.scale_stu),
        T.ColorJitter(brightness=args.color_stu, contrast=args.color_stu, saturation=args.color_stu),
        T.GaussianBlur(high=args.blur_stu),
        T.ToTensor(),
        normalize
    ])
    # if args.test == 2:
    #     tgt_train_transform_tea = T.Compose([
    #         T.RandomAffineRotation(args.rotation_tea, args.shear_tea, args.translate_tea, args.scale_tea),
    #         T.ToTensor(),
    #         normalize
    #     ])
    # else:
    tgt_train_transform_tea = T.Compose([
        T.RandomAffineRotation(args.rotation_tea, args.shear_tea, args.translate_tea, args.scale_tea),
        T.ColorJitter(brightness=args.color_tea, contrast=args.color_tea, saturation=args.color_tea),
        T.GaussianBlur(high=args.blur_tea),
        T.ToTensor(),
        normalize
        ])
    val_transform = T.Compose([
        T.Resize(args.image_size),
        T.ToTensor(),
        normalize
    ])
    image_size = (args.image_size, args.image_size)
    heatmap_size = (args.heatmap_size, args.heatmap_size)

    return src_train_transform, base_transform, tgt_train_transform_ori_stu, tgt_train_transform_ori_tea, tgt_train_transform_stu, tgt_train_transform_tea, val_transform, image_size, heatmap_size, label_transform


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log + '_' + args.arch, args.phase)

    logger.write(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False     
        # warnings.warn('You have chosen to seed training. '
        #               'This will turn on the CUDNN deterministic setting, '
        #               'which can slow down your training considerably! '
        #               'You may see unexpected behavior when restarting '
        #               'from checkpoints.')

    cudnn.benchmark = True
    src_train_transform, base_transform, tgt_train_transform_ori_stu, tgt_train_transform_ori_tea, tgt_train_transform_stu, tgt_train_transform_tea, val_transform, image_size, heatmap_size, label_transform = img_trans()
    source_dataset = datasets.__dict__[args.source]
    train_source_dataset = source_dataset(root=args.source_root, transforms=src_train_transform,
                                          image_size=image_size, heatmap_size=heatmap_size)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_source_dataset = source_dataset(root=args.source_root, split='test', transforms=val_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
    val_source_loader = DataLoader(val_source_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True, drop_last=True)
    # create model

    student = models.__dict__[args.arch](num_keypoints=train_source_dataset.num_keypoints).cuda()
    teacher = models.__dict__[args.arch](num_keypoints=train_source_dataset.num_keypoints).cuda()


    student = torch.nn.DataParallel(student).cuda()
    teacher = torch.nn.DataParallel(teacher).cuda()

  
    pretrained_dict = torch.load('/DATA/pengbaichao/logs/human/source_vit_model_dino/final.pt', map_location='cpu')['student']
    model_dict = student.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    student.load_state_dict(pretrained_dict, strict=False)
    teacher.load_state_dict(pretrained_dict, strict=False)

    target_dataset = datasets.__dict__[args.target_train]
    train_target_dataset = target_dataset(root=args.target_root, transforms_base=base_transform, transforms_ori_stu=tgt_train_transform_ori_stu, transforms_ori_tea=tgt_train_transform_ori_tea, 
                                        transforms_stu=tgt_train_transform_stu, transforms_tea=tgt_train_transform_tea, 
                                        k=args.k, image_size=image_size, heatmap_size=heatmap_size)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    
    target_dataset = datasets.__dict__[args.target]
    val_target_dataset = target_dataset(root=args.target_root, split='test', transforms=val_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
    val_target_loader = DataLoader(val_target_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True, drop_last=True)



    logger.write("Target train: {}".format(len(train_target_loader)))
    # logger.write("Source test: {}".format(len(val_source_loader)))
    logger.write("Target test: {}".format(len(val_target_loader)))

    # train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)



    criterion = JointsMSELoss()
    con_criterion = ConsLoss()

    stu_optimizer = AdamW(student.parameters(), lr=args.lr)
    tea_optimizer = AdamW(teacher.parameters(), lr=args.lr*0.4)
    
    EMA_optimizer = OldWeightEMA(teacher, student, alpha=args.teacher_alpha)

    lr_scheduler = MultiStepLR(stu_optimizer, args.lr_step, args.lr_factor)


    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        student.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        stu_optimizer.load_state_dict(checkpoint['stu_optimizer'])
        # tea_optimizer.load_state_dict(checkpoint['tea_optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    # define visualization function
    tensor_to_image = Compose([
        Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToPILImage()
    ])

    def visualize(image, keypoint2d, name):
        """
        Args:
            image (tensor): image in shape 3 x H x W
            keypoint2d (tensor): keypoints in shape K x 2
            name: name of the saving image
        """
        train_source_dataset.visualize(tensor_to_image(image),
                                       keypoint2d, logger.get_image_path("{}.jpg".format(name)))


    if args.vsl:
        save_vsl(val_target_loader, teacher, criterion, visualize if args.debug else None, args)    # if args.phase == 'test':
        exit(0)
    logger.write("*********************Source Only**************************")
    # source_val_acc = validate(val_source_loader, teacher, criterion, None, args)
    target_val_acc = validate(val_target_loader, teacher, criterion, visualize, args)

    # logger.write("Source: {:4.3f} Target: {:4.3f}".format(source_val_acc['all'], target_val_acc['all']))
    logger.write("Target: {:4.3f}".format(target_val_acc['all']))
    for name, acc in target_val_acc.items():
        logger.write("{}: {:4.3f}".format(name, acc))
    logger.write(("***********************************************************"))

    # start training
    best_acc = 0
    start_epoch = 0
    # args.iter_count=0
    # args.interval = int(args.iters_per_epoch * args.epochs / (args.rounds+1))
    for epoch in range(start_epoch, args.epochs):
    # while args.iter_count < args.iters_per_epoch * args.epochs:
        # epoch = args.iter_count
        logger.set_epoch(epoch)
        lr_scheduler.step()
        train(train_target_iter,  student, teacher,  criterion, con_criterion, 
                stu_optimizer, EMA_optimizer, tea_optimizer, epoch,visualize if args.debug else None, args)


        # evaluate on validation set
        source_val_acc = validate(val_source_loader, teacher, criterion, visualize, args)
        target_val_acc = validate(val_target_loader, teacher, criterion, visualize if args.debug else None, args)

        if target_val_acc['all'] > best_acc:
            torch.save(
                {
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'stu_optimizer': stu_optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args
                }, logger.get_checkpoint_path('best_pt')
            )
            best_acc = target_val_acc['all']
        torch.save(
            {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'stu_optimizer': stu_optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }, logger.get_checkpoint_path('final_pt')
        )

        # logger.write("Epoch: {} Source: {:4.3f} Target: {:4.3f} Target(best): {:4.3f}".format(epoch, source_val_acc['all'], target_val_acc['all'], best_acc))
        logger.write("Epoch: {} Source: {:4.3f} Target: {:4.3f} Target(best): {:4.3f}".format(epoch, source_val_acc['all'], target_val_acc['all'], best_acc))
        for name, acc in target_val_acc.items():
            logger.write("{}: {:4.3f}".format(name, acc))

    logger.close()

def train(train_target_iter,  student, teacher,  criterion, con_criterion,
          stu_optimizer, EMA_optimizer, tea_optimizer, epoch: int, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_all = AverageMeter('Loss (all)', ":.4e")
    losses_p = AverageMeter('Loss (p)', ":.4e")
    losses_c = AverageMeter('Loss (c)', ":.4e")
    # acc_s = AverageMeter("Acc (s)", ":3.2f")

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_all, losses_p, losses_c],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student.train()
    teacher.train()

    end = time.time()
    scaler = torch.cuda.amp.GradScaler()
    # marks = get_simple_hard_samples(student=student, teacher=teacher, iter=train_target_iter, args=args)
    for j in tqdm(range(args.iters_per_epoch)):

        stu_optimizer.zero_grad()

        # x_s, label_s, weight_s, meta_s = next(train_source_iter)
        # x_s, label_s, w, meta_s, _, _, _, _,  img_lb= next(train_source_iter)

        # weight_s = torch.tensor(np.array([1.] * 6 + [0, 0] + [1.] * 8, dtype=np.float32)).to(device)
        # weight_s = weight_s.unsqueeze(0).repeat(x_s.size(0),1).unsqueeze(2)
        # print(weight_s)
        # print(weight_s.size())
        
        # print(weight_s)
        # label_s = label_generate(x_s, source_model)
        x_t_ori_stu, x_t_ori_tea, x_t_stu, _, _, meta_t_stu, x_t_teas, _, _, meta_t_tea = next(train_target_iter)

        # x_s = x_s.to(device)
        # x_s_ori = x_s.clone()
        x_t_ori_stu = x_t_ori_stu.to(device)
        x_t_ori_tea = x_t_ori_tea.to(device)
        x_t_stu = x_t_stu.to(device)
        x_t_teas = [x_t_tea.to(device) for x_t_tea in x_t_teas]
        x_t_teas_ori = [x_t_tea.clone() for x_t_tea in x_t_teas]
        # label_s = label_s.to(device)
        # real_lbs = real_lbs.to(device)
        # weight_s = weight_s.to(device)
        label_t = meta_t_stu['target_ori'].to(device)
        weight_t = meta_t_stu['target_weight_ori'].to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        ratio = args.image_size / args.heatmap_size

        with torch.no_grad():

            y_t_teas = [teacher(x_t_tea) for x_t_tea in x_t_teas] # softmax on w, h
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
                tea_mask[ind] = 1.

            angle, [trans_x, trans_y], [shear_x, shear_y], scale = meta_t_stu['aug_param_stu']

        with torch.cuda.amp.autocast():

            y_t_stu, _ = student(x_t_stu) # softmax on w, h
            y_t_stu_recon = torch.zeros_like(y_t_stu).cuda() # b, c, h, w
            for ind in range(x_t_stu.size(0)):
                _angle, _trans_x, _trans_y, _shear_x, _shear_y, _scale = angle[ind].item(), trans_x[ind].item(), trans_y[ind].item(), shear_x[ind].item(), shear_y[ind].item(), scale[ind].item()
                temp = tF.affine(y_t_stu[ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                temp = tF.affine(temp, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                y_t_stu_recon[ind] = tF.affine(temp, 0., translate=[0., 0.], shear=[_shear_x, _shear_y], scale=1.)
            activates = y_t_tea_recon.amax(dim=(2,3))
            y_t_tea_recon = rectify(y_t_tea_recon, sigma=args.sigma)
            mask_thresh = torch.kthvalue(activates.view(-1), int(args.mask_ratio * activates.numel()))[0].item()
            tea_mask = tea_mask * activates>mask_thresh

            # the singal 1 denotes the sample is naive
            marks = getmark(y_t_stu_recon, y_t_tea_recon)

            x_t_stu_occlude = torch.zeros_like(x_t_stu)
            for index in range(args.batch_size):
                if marks[index]==0:
                    x_t_stu_occlude[index] = random_mask_tensor_images(x_t_ori_stu[index], mask_size=(80, 80))
                else:
                    x_t_stu_occlude[index] = random_mask_tensor_images(splice_images_random(x_t_ori_stu[index], x_t_ori_tea[index]),mask_size=(80, 80))
            y_t_stu_occlude, _ = student(x_t_stu_occlude)

            # if args.mix_type == 'full':
            #     rho = np.random.beta(0.75, 0.75)
            #     rand_ind = torch.randperm(x_t_stu.size(0))
            #     x_t_stu_rand = x_t_stu[rand_ind]
            #     y_t_stu_rand = y_t_stu[rand_ind]
            #     mix_x_t_stu = x_t_stu * rho + x_t_stu_rand * (1-rho)
            #     mix_y_t_stu = student(mix_x_t_stu)
            #     target_mix_y_t_stu = y_t_stu * rho + y_t_stu_rand * (1-rho)

            #     loss_m = criterion(mix_y_t_stu, target_mix_y_t_stu.detach())




            loss_p = con_criterion(y_t_stu_occlude, y_t_stu_recon, tea_mask=tea_mask)   
            loss_c = con_criterion(y_t_stu_recon, y_t_tea_recon, tea_mask=tea_mask)
        loss_all = args.lambda_c * loss_c + loss_p
        scaler.scale(loss_all).backward()
        scaler.step(stu_optimizer)
        scaler.step(tea_optimizer)
        EMA_optimizer.step()
        scaler.update()
        losses_all.update(loss_all, x_t_stu.size(0))
        losses_c.update(loss_c, x_t_stu.size(0))
        losses_p.update(loss_p, x_t_stu.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if j % args.print_freq == 0:
            progress.display(j)
            # if visualize is not None:
            #     visualize(x_s[0], pred_s[0] * args.image_size / args.heatmap_size, "proxy_{}_pred.jpg".format(i))
            #     visualize(x_s[0], meta_s['keypoint2d_stu'][0], "proxy_{}_pseudolabel.jpg".format(i))


def validate(val_loader, model, criterion, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    acc = AverageMeterList(list(range(val_loader.dataset.num_keypoints)), ":3.2f",  ignore_val=-1)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses], 
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x, label, weight, meta) in enumerate(val_loader):
            x = x.to(device)
            label = label.to(device)
            weight = weight.to(device)

            # compute output
            y, _ = model(x)
            loss = criterion(y, label, weight)

            # measure accuracy and record loss
            losses.update(loss.item(), x.size(0))
            acc_per_points, avg_acc, cnt, pred = accuracy(y.cpu().numpy(),
                                                          label.cpu().numpy())
            acc.update(acc_per_points, x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.val_print_freq == 0:
                progress.display(i)
                if visualize is not None:
                    visualize(x[0], pred[0] * args.image_size / args.heatmap_size, "val_{}_pred.jpg".format(i))
                    visualize(x[0], meta['keypoint2d'][0], "val_{}_label.jpg".format(i))

    return val_loader.dataset.group_accuracy(acc.average())

def save_vsl(val_loader, model, criterion, visualize, args: argparse.Namespace):

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # end = time.time()
        for i, (x, label, weight, meta) in enumerate(val_loader):
            x = x.to(device)
            label = label.to(device)
            weight = weight.to(device)

            # compute output
            y, _ = model(x)
            # loss = criterion(y, label, weight)

            # measure accuracy and record loss
            # losses.update(loss.item(), x.size(0))
            acc_per_points, avg_acc, cnt, pred = accuracy(y.cpu().numpy(),
                                                          label.cpu().numpy())
            # acc.update(acc_per_points, x.size(0))

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            # if i % args.val_print_freq == 0:
            # progress.display(i)
                # if visualize is not None:
            for j in range(x.size(0)):
                visualize(x[j], pred[j] * args.image_size / args.heatmap_size, "val_{}_{}_pred.jpg".format(i,j))
                visualize(x[j], meta['keypoint2d'][j], "val_{}_{}_label.jpg".format(i,j))

    # return val_loader.dataset.group_accuracy(acc.average())

if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='Source Only for Keypoint Detection Domain Adaptation')
    # dataset parameters
    parser.add_argument('--source_root', default='/DATA/pengbaichao/DATA_pose/data/surreal_data/human_data/train/', help='root path of the source dataset')
    parser.add_argument('--target_root', default='/DATA/pengbaichao/DATA_pose/data/surreal_data/LSP/', help='root path of the target dataset')
    parser.add_argument('-s', '--source', default='SURREAL', help='source domain(s)')
    parser.add_argument('-t', '--target', default='LSP', help='target domain(s)')
    parser.add_argument('--proxy', default='proxy', help='source domain(s)')
    parser.add_argument('--target-train', default='LSP_mt', help='target domain(s)')
    parser.add_argument('--resize-scale', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--image-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--heatmap-size', type=int, default=64,
                        help='output heatmap size')
    parser.add_argument('--sigma', type=int, default=2,
                        help='')
    parser.add_argument('--k', type=int, default=1,
                        help='')
    
    # augmentation
    parser.add_argument('--rotation_stu', type=int, default=60,
                        help='rotation range of the RandomRotation augmentation')
    parser.add_argument('--color_stu', type=float, default=0.25,
                        help='color range of the jitter augmentation')
    parser.add_argument('--blur_stu', type=float, default=0,
                        help='blur range of the jitter augmentation')
    parser.add_argument('--shear_stu', nargs='+', type=float, default=(-30, 30),
                        help='shear range for the RandomResizeCrop augmentation')
    parser.add_argument('--translate_stu', nargs='+', type=float, default=(0.05, 0.05),
                        help='tranlate range for the RandomResizeCrop augmentation')
    parser.add_argument('--scale_stu', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--rotation_tea', type=int, default=60,
                        help='rotation range of the RandomRotation augmentation')
    parser.add_argument('--color_tea', type=float, default=0.25,
                        help='color range of the jitter augmentation')
    parser.add_argument('--blur_tea', type=float, default=0,
                        help='blur range of the jitter augmentation')
    parser.add_argument('--shear_tea', nargs='+', type=float, default=(-30, 30),
                        help='shear range for the RandomResizeCrop augmentation')
    parser.add_argument('--translate_tea', nargs='+', type=float, default=(0.05, 0.05),
                        help='tranlate range for the RandomResizeCrop augmentation')
    parser.add_argument('--scale_tea', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--s2t-freq', type=float, default=0.5)
    parser.add_argument('--s2t-alpha', nargs='+', type=float, default=(0, 1))
    parser.add_argument('--t2s-freq', type=float, default=0.5)
    parser.add_argument('--t2s-alpha', nargs='+', type=float, default=(0, 1))

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='pose_resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: pose_resnet101)')

    parser.add_argument("--resume", type=str, default=None,
                        help="where restore model parameters from.")
    parser.add_argument("--pretrain", type=str, default=None,
                        help="where restore model parameters from.")

    # training parameters
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--test-batch', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.00004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--teacher_alpha', default=0.9995, type=float)
    parser.add_argument('--lr-step', default=[7], type=tuple, help='parameter for lr scheduler')
    parser.add_argument('--lr-factor', default=0.5, type=float, help='parameter for lr scheduler')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--val-print-freq', default=2000, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='test',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--debug', default=1, type=float,
                        help='In the debug mode, save images and predictions')
    parser.add_argument('--mask-ratio', type=float, default=0.5,
                        help='')

    parser.add_argument('--lambda_c', default=1., type=float)
    parser.add_argument('--mix_ratio', type=float, default=1)
    parser.add_argument('--proxy_ratio', type=float, default=1)
    parser.add_argument('--pct', type=float, default=0.1)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--step_len', type=float, default=0.1)

    parser.add_argument('--resplit', type=str, default='spl')
    parser.add_argument('--split_way', default='mse', type=str)
    parser.add_argument('--mix_type', default='full', type=str)
    parser.add_argument('--init_split', action="store_true")
    parser.add_argument('--vsl', action="store_true")
    parser.add_argument("--finetune_out", type=str, default='/DATA/pengbaichao/checkpoint/human')
    parser.add_argument('--source_ckpt', default=None, type=str)

    args = parser.parse_args()
    args.log = 'logs/S2L/syn2real_'
    main(args)