import scipy.io as scio
import os

from PIL import ImageFile
import torch
from .keypoint_dataset import Body16KeypointDataset
from ..transforms.keypoint_detection import *
from .util import *
from ._util import download as download_data, check_exits
import torchvision.transforms.functional as tF


ImageFile.LOAD_TRUNCATED_IMAGES = True


class LSP_mt_proxy(Body16KeypointDataset):
    """`Leeds Sports Pose Dataset <http://sam.johnson.io/research/lsp.html>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): PlaceHolder.
        task (str, optional): Placeholder.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transforms (callable, optional): PlaceHolder.
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    .. note:: In `root`, there will exist following files after downloading.
        ::
            lsp/
                images/
                joints.mat

    .. note::
        LSP is only used for target domain. Due to the small dataset size, the whole dataset is used
        no matter what ``split`` is. Also, the transform is fixed.
    """
    def __init__(self, root, tgt_list=None, split='train', task='all', download=True, image_size=(256, 256), k=1, 
                    transforms_base=None, transforms_stu=None, transforms_tea=None, **kwargs):
        if download:
            download_data(root, "images", "lsp_dataset.zip",
                          "https://cloud.tsinghua.edu.cn/f/46ea73c89abc46bfb125/?dl=1")
        else:
            check_exits(root, "lsp")

        assert split in ['train', 'test', 'all']
        self.split = split
        self.transforms_base = Compose([ResizePad(image_size[0])]) + transforms_base
        self.transforms_stu = transforms_stu
        self.transforms_tea = transforms_tea
        # self.reverse_target = reverse_target
        self.k = k
        if tgt_list is not None:
            samples=tgt_list
        else:
            samples = []
            annotations = scio.loadmat(os.path.join(root, "joints.mat"))['joints'].transpose((2, 1, 0))
            for i in range(0, 2000):
                image = "im{0:04d}.jpg".format(i+1)
                annotation = annotations[i]
                samples.append((image, annotation))

        self.joints_index = (0, 1, 2, 3, 4, 5, 13, 13, 12, 13, 6, 7, 8, 9, 10, 11)
        self.visible = np.array([1.] * 6 + [0, 0] + [1.] * 8, dtype=np.float32)

        super(LSP_mt_proxy, self).__init__(root, samples, image_size=image_size, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample[0]
        image = Image.open(os.path.join(self.root, "images", image_name))
        keypoint2d = sample[1][self.joints_index, :2]
        image, data = self.transforms_base(image, keypoint2d=keypoint2d, intrinsic_matrix=None)
        keypoint2d = data['keypoint2d']

        image_stu, data_stu = self.transforms_stu(image, keypoint2d=keypoint2d, intrinsic_matrix=None)
        keypoint2d_stu = data_stu['keypoint2d']
        aug_param_stu = data_stu['aug_param']

        visible = self.visible * (1-sample[1][self.joints_index, 2])
        visible = visible[:, np.newaxis]

        # 2D heatmap
        target_stu, target_weight_stu = generate_target(keypoint2d_stu, visible, self.heatmap_size, self.sigma, self.image_size)
        target_stu = torch.from_numpy(target_stu)
        target_weight_stu = torch.from_numpy(target_weight_stu)

        target_ori, target_weight_ori = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target_ori = torch.from_numpy(target_ori)
        target_weight_ori = torch.from_numpy(target_weight_ori)

        meta_stu = {
            'image': image_name,
            'target_small_stu': generate_target(keypoint2d_stu, visible, (8, 8), self.sigma, self.image_size),
            'keypoint2d_ori': keypoint2d,  
            'target_ori': target_ori,  
            'target_weight_ori': target_weight_ori,  
            'keypoint2d_stu': keypoint2d_stu,  # （NUM_KEYPOINTS x 2）
            'aug_param_stu': aug_param_stu,
        }

        crop_size = 224
        start = (256 - crop_size) // 2
        end = start + crop_size

        # 裁剪图像Tensor
        image_stu = image_stu[:, start:end, start:end]
        # images_tea[0] = images_tea[0][:, start:end, start:end]

        return image_stu, target_stu, target_weight_stu, meta_stu
