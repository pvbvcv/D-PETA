import os
import json
import tqdm
from PIL import ImageFile
import torch
from .keypoint_dataset import Body16KeypointDataset
from ..transforms.keypoint_detection import *
from .util import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Human36M_mt(Body16KeypointDataset):
    """`Human3.6M Dataset <http://vision.imar.ro/human3.6m/description.php>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, ``test``, or ``all``.
            Default: ``train``.
        task (str, optional): Placeholder.
        download (bool, optional): Placeholder.
        transforms (callable, optional): A function/transform that takes in a dict (which contains PIL image and
            its labels) and returns a transformed version. E.g, :class:`~common.vision.transforms.keypoint_detection.Resize`.
        image_size (tuple): (width, height) of the image. Default: (256, 256)
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    .. note:: You need to download Human36M manually.
        Ensure that there exist following files in the `root` directory before you using this class.
        ::
            annotations/
                Human36M_subject11_joint_3d.json
                ...
            images/

    .. note::
        We found that the original Human3.6M image is in high resolution while most part in an image is background,
        thus we crop the image and keep only the surrounding area of hands (1.5x bigger than hands) to speed up training.
        In `root`, there will exist following files after crop.
        ::
            Human36M_crop/
            annotations/
                keypoints2d_11.json
                ...
    """
    def __init__(self, root, split='train', task='all', download=False, k=1, 
                    transforms_base=None, transforms_stu=None, transforms_tea=None, **kwargs):
        assert split in ['train', 'test', 'all']
        self.split = split
        self.transforms_base = transforms_base
        self.transforms_stu = transforms_stu
        self.transforms_tea = transforms_tea
        self.k = k

        samples = []
        if self.split == 'train':
            parts = [1, 5, 6, 7, 8]
        elif self.split == 'test':
            parts = [9, 11]
        else:
            parts = [1, 5, 6, 7, 8, 9, 11]

        for part in parts:
            annotation_file = os.path.join(root, 'annotations/keypoints2d_{}.json'.format(part))
            if not os.path.exists(annotation_file):
                self.preprocess(part, root)
            print("loading", annotation_file)
            with open(annotation_file) as f:
                samples.extend(json.load(f))
        # decrease the number of test samples to decrease the time spent on test
        random.seed(42)
        if self.split == 'test':
            samples = random.choices(samples, k=3200)
        super(Human36M_mt, self).__init__(root, samples, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = os.path.join(self.root, "crop_images", image_name)
        image = Image.open(image_path)
        keypoint3d_camera = np.array(sample['keypoint3d'])  # NUM_KEYPOINTS x 3
        keypoint2d = np.array(sample['keypoint2d'])  # NUM_KEYPOINTS x 2
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]

        image, data = self.transforms_base(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d = data['keypoint2d']
        intrinsic_matrix = data['intrinsic_matrix']
        keypoint3d_camera = keypoint2d_to_3d(keypoint2d, intrinsic_matrix, Zc)

        image_stu, data_stu = self.transforms_stu(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d_stu = data_stu['keypoint2d']
        intrinsic_matrix_stu = data_stu['intrinsic_matrix']
        aug_param_stu = data_stu['aug_param']
        keypoint3d_camera_stu = keypoint2d_to_3d(keypoint2d_stu, intrinsic_matrix_stu, Zc)

        # noramlize 2D pose:
        visible = np.ones((self.num_keypoints, ), dtype=np.float32)
        visible = visible[:, np.newaxis]

        target_stu, target_weight_stu = generate_target(keypoint2d_stu, visible, self.heatmap_size, self.sigma, self.image_size)
        target_stu = torch.from_numpy(target_stu)
        target_weight_stu = torch.from_numpy(target_weight_stu)
        
        target_ori, target_weight_ori = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target_ori = torch.from_numpy(target_ori)
        target_weight_ori = torch.from_numpy(target_weight_ori)

        # normalize 3D pose:
        # put middle finger metacarpophalangeal (MCP) joint in the center of the coordinate system
        # and make distance between wrist and middle finger MCP joint to be of length 1
        keypoint3d_n_stu = keypoint3d_camera_stu - keypoint3d_camera_stu[9:10, :]
        keypoint3d_n_stu = keypoint3d_n_stu / np.sqrt(np.sum(keypoint3d_n_stu[0, :] ** 2))
        z_stu = keypoint3d_n_stu[:, 2]
        meta_stu = {
            'image': image_name,
            'target_small_stu': generate_target(keypoint2d_stu, visible, (8, 8), self.sigma, self.image_size),
            'keypoint2d_ori': keypoint2d,  
            'target_ori': target_ori,  
            'target_weight_ori': target_weight_ori,  
            'keypoint2d_stu': keypoint2d_stu,  # （NUM_KEYPOINTS x 2）
            'keypoint3d_stu': keypoint3d_n_stu,  # （NUM_KEYPOINTS x 3）
            'aug_param_stu': aug_param_stu,
            'z_stu': z_stu,
        }

        images_tea, targets_tea, target_weights_tea, metas_tea = [], [], [], []
        for _ in range(self.k):
            image_tea, data_tea = self.transforms_tea(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
            keypoint2d_tea = data_tea['keypoint2d']
            intrinsic_matrix_tea = data_tea['intrinsic_matrix']
            aug_param_tea = data_tea['aug_param']
            keypoint3d_camera_tea = keypoint2d_to_3d(keypoint2d_tea, intrinsic_matrix_tea, Zc)

            # 2D heatmap
            target_tea, target_weight_tea = generate_target(keypoint2d_tea, visible, self.heatmap_size, self.sigma, self.image_size)
            target_tea = torch.from_numpy(target_tea)
            target_weight_tea = torch.from_numpy(target_weight_tea)

            # normalize 3D pose:
            # put middle finger metacarpophalangeal (MCP) joint in the center of the coordinate system
            # and make distance between wrist and middle finger MCP joint to be of length 1
            keypoint3d_n_tea = keypoint3d_camera_tea - keypoint3d_camera_tea[9:10, :]
            keypoint3d_n_tea = keypoint3d_n_tea / np.sqrt(np.sum(keypoint3d_n_tea[0, :] ** 2))
            z_tea = keypoint3d_n_tea[:, 2]

            meta_tea = {
                'image': image_name,
                'target_small_tea': generate_target(keypoint2d_tea, visible, (8, 8), self.sigma, self.image_size),
                'keypoint2d_tea': keypoint2d_tea,  # （NUM_KEYPOINTS x 2）
                'keypoint3d_tea': keypoint3d_n_tea,  # （NUM_KEYPOINTS x 3）
                'aug_param_tea': aug_param_tea,
                'z_tea': z_tea,
            }

            images_tea.append(image_tea)
            targets_tea.append(target_tea)
            target_weights_tea.append(target_weight_tea)
            metas_tea.append(meta_tea)

        return image_stu, target_stu, target_weight_stu, meta_stu, images_tea, targets_tea, target_weights_tea, metas_tea

    def preprocess(self, part, root):
        body_index = [3, 2, 1, 4, 5, 6, 0, 11, 8, 10, 16, 15, 14, 11, 12, 13]
        image_size = 512
        print("preprocessing part", part)
        camera_json = os.path.join(root, "annotations", "Human36M_subject{}_camera.json".format(part))
        data_json = os.path.join(root, "annotations", "Human36M_subject{}_data.json".format(part))
        joint_3d_json = os.path.join(root, "annotations", "Human36M_subject{}_joint_3d.json".format(part))
        with open(camera_json, "r") as f:
            cameras = json.load(f)
        with open(data_json, "r") as f:
            data = json.load(f)
            images = data['images']

        with open(joint_3d_json, "r") as f:
            joints_3d = json.load(f)

        data = []

        for i, image_data in enumerate(tqdm.tqdm(images)):
            # downsample
            if i % 5 == 0:
                keypoint3d = np.array(joints_3d[str(image_data["action_idx"])][str(image_data["subaction_idx"])][
                                          str(image_data["frame_idx"])])
                keypoint3d = keypoint3d[body_index, :]
                keypoint3d[7, :] = 0.5 * (keypoint3d[12, :] + keypoint3d[13, :])
                camera = cameras[str(image_data["cam_idx"])]
                R, T = np.array(camera["R"]), np.array(camera['t'])[:, np.newaxis]
                extrinsic_matrix = np.concatenate([R, T], axis=1)
                keypoint3d_camera = np.matmul(extrinsic_matrix, np.hstack(
                    (keypoint3d, np.ones((keypoint3d.shape[0], 1)))).T)  # (3 x NUM_KEYPOINTS)
                Z_c = keypoint3d_camera[2:3, :]  # 1 x NUM_KEYPOINTS

                f, c = np.array(camera["f"]), np.array(camera['c'])
                intrinsic_matrix = np.zeros((3, 3))
                intrinsic_matrix[0, 0] = f[0]
                intrinsic_matrix[1, 1] = f[1]
                intrinsic_matrix[0, 2] = c[0]
                intrinsic_matrix[1, 2] = c[1]
                intrinsic_matrix[2, 2] = 1
                keypoint2d = np.matmul(intrinsic_matrix, keypoint3d_camera)  # (3 x NUM_KEYPOINTS)
                keypoint2d = keypoint2d[0: 2, :] / Z_c
                keypoint2d = keypoint2d.T
                src_image_path = os.path.join(root, "images", image_data['file_name'])
                tgt_image_path = os.path.join(root, "crop_images", image_data['file_name'])
                os.makedirs(os.path.dirname(tgt_image_path), exist_ok=True)
                image = Image.open(src_image_path)

                bounding_box = get_bounding_box(keypoint2d)
                w, h = image.size
                left, upper, right, lower = scale_box(bounding_box, w, h, 1.5)
                image, keypoint2d = crop(image, upper, left, lower-upper+1, right-left+1, keypoint2d)
                Z_c = Z_c.T

                # Calculate XYZ from uvz
                uv1 = np.concatenate([np.copy(keypoint2d), np.ones((16, 1))],
                                     axis=1)  # NUM_KEYPOINTS x 3
                uv1 = uv1 * Z_c  # NUM_KEYPOINTS x 3
                keypoint3d_camera = np.matmul(np.linalg.inv(intrinsic_matrix), uv1.T).T

                # resize image will change camera intrinsic matrix
                w, h = image.size
                image = image.resize((image_size, image_size))
                image.save(tgt_image_path)

                zoom_factor = float(w) / float(image_size)
                keypoint2d /= zoom_factor
                intrinsic_matrix[0, 0] /= zoom_factor
                intrinsic_matrix[1, 1] /= zoom_factor
                intrinsic_matrix[0, 2] /= zoom_factor
                intrinsic_matrix[1, 2] /= zoom_factor

                data.append({
                    "name": image_data['file_name'],
                    'keypoint2d': keypoint2d.tolist(),
                    'keypoint3d': keypoint3d_camera.tolist(),
                    'intrinsic_matrix': intrinsic_matrix.tolist(),
                })

        with open(os.path.join(root, "annotations", "keypoints2d_{}.json".format(part)), "w") as f:
            json.dump(data, f)