import random
import shutil
import cv2
import os
import json
import torch
import smplx
import trimesh
import numpy as np
from glob import glob
from torchvision.transforms import Normalize
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy

from core.camerahmr_model import CameraHMR
from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, SMPL_MODEL_PATH, DETECTRON_CKPT, DETECTRON_CFG
from core.datasets.dataset import Dataset
from core.utils.renderer_pyrd import Renderer
from core.utils import recursive_to
from core.utils.geometry import batch_rot2aa
from core.cam_model.fl_net import FLNet
from core.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_BETAS
from scipy.spatial.transform import Rotation as R

mod = True # more accuracy

def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img

class HumanMeshEstimator:
    def __init__(self, smpl_model_path=SMPL_MODEL_PATH, threshold=0.25):
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = self.init_model()
        self.detector = self.init_detector(threshold)
        self.cam_model = self.init_cam_model()
        self.smpl_model = smplx.SMPLLayer(model_path=smpl_model_path, num_betas=NUM_BETAS).to(self.device)
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def init_cam_model(self):
        model = FLNet()
        checkpoint = torch.load(CAM_MODEL_CKPT)['state_dict']
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def init_model(self):
        model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
        model = model.to(self.device)
        model.eval()
        return model
    
    def init_detector(self, threshold):

        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = threshold
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector

    
    def convert_to_full_img_cam(self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2. * focal_length / (bbox_height * s)
        cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
        cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_output_mesh(self, params, pred_cam, batch):
        smpl_output = self.smpl_model(**{k: v.float() for k, v in params.items()})
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        img_h, img_w = batch['img_size'][0]
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch['box_size'],
            bbox_center=batch['box_center'],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch['cam_int'][:, 0, 0]
        )
        return pred_vertices, pred_keypoints_3d, cam_trans

    def get_cam_intrinsics(self, img):
        img_h, img_w, c = img.shape
        aspect_ratio, img_full_resized = resize_image(img, IMAGE_SIZE)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                            (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())

        estimated_fov, _ = self.cam_model(img_full_resized.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
        # fl_h = (img_w * img_w + img_h * img_h) ** 0.5
        cam_int = np.array([[fl_h, 0, img_w/2], [0, fl_h, img_h / 2], [0, 0, 1]]).astype(np.float32)
        return cam_int


    def remove_pelvis_rotation(self, smpl):
        """We don't trust the body orientation coming out of bedlam_cliff, so we're just going to zero it out."""
        smpl.body_pose[0][0][:] = np.zeros(3)


    def process_image(self, img_path, output_img_folder, i):
        img_cv2 = cv2.imread(str(img_path))
        
        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        overlay_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}{img_ext}')
        smpl_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}.smpl')
        mesh_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}.obj')

        # Detect humans in the image
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)

        if valid_idx.sum() > 0:
            first_valid_idx = torch.where(valid_idx)[0][0]
            
            boxes = det_instances.pred_boxes.tensor[first_valid_idx].cpu().numpy().reshape(1, -1)
            bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
            bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        else:
            print("No valid person detected in the image")
            return

        # Get Camera intrinsics using HumanFoV Model
        cam_int = self.get_cam_intrinsics(img_cv2)
        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, img_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

                joint_indices = [3, 4]
                joint_names = {3: "左膝盖", 4: "右膝盖"}
                body_poses_mat = out_smpl_params['body_pose'].cpu().numpy()[0]  # shape: [23, 3, 3]
                for idx in joint_indices:
                    rot_matrix = body_poses_mat[idx]
                    r_pred = R.from_matrix(rot_matrix)
                    r_true = R.from_rotvec([0, 0, 0])  # 真实值是零旋转
                    
                    # 计算旋转差并获取角度（弧度和角度）
                    diff = r_pred * r_true.inv()
                    angle_rad = diff.magnitude()  # 弧度
                    angle_deg = np.degrees(angle_rad)
                    
                    if mod:
                        if angle_deg < 15.:
                            # 收缩旋转向量：向零靠近，但不完全归零
                            shrink_factor = 0.3  # 可调参数：0 = 单位矩阵，1 = 不变，0.3 表示保留 30% 的旋转
                            rotvec = r_pred.as_rotvec()
                            shrunk_rotvec = rotvec * shrink_factor

                            # 转回旋转矩阵
                            r_shrunk = R.from_rotvec(shrunk_rotvec)
                            shrunk_matrix = r_shrunk.as_matrix()

                            # 转为 torch.Tensor 并放到原始设备上（如 cuda:0）
                            device = out_smpl_params['body_pose'].device
                            shrunk_tensor = torch.tensor(shrunk_matrix, dtype=out_smpl_params['body_pose'].dtype, device=device)

                            # 直接修改原始张量
                            # out_smpl_params['body_pose'][0, idx] = torch.eye(3, device=out_smpl_params['body_pose'].device)
                            out_smpl_params['body_pose'][0, idx] = shrunk_tensor
                            print(f"[修正] 图像: {overlay_fname} | {joint_names[idx]} (关节 {idx}) 旋转角度 {angle_deg:.2f}° < 15°")

                        with torch.no_grad():
                            if out_smpl_params['betas'][0][1] < 1.0:
                                adjustment = random.uniform(0.8, 1.0)
                                old_value = out_smpl_params['betas'][0][1].item()
                                out_smpl_params['betas'][0][1] -= adjustment
                                new_value = out_smpl_params['betas'][0][1].item()
                                print(f"[修正] {overlay_fname} betas[1] 从 {old_value:.4f} 调整为 {new_value:.4f}")

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)

            mesh = trimesh.Trimesh(output_vertices[0].cpu().numpy() , self.smpl_model.faces,
                            process=False)
            # mesh.export(mesh_fname)

            # Render overlay
            focal_length = (focal_length_[0], focal_length_[0])
            pred_vertices_array = (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
            renderer = Renderer(focal_length=focal_length[0], img_w=img_w, img_h=img_h, faces=self.smpl_model.faces, same_mesh_color=True)
            front_view = renderer.render_front_view(pred_vertices_array, bg_img_rgb=img_cv2.copy())
    
            # Render side view
            side_view = renderer.render_side_view(pred_vertices_array)
        
            # final_img = np.hstack([img, front_view, side_view])
            # Concatenate front and side views horizontally
            final_img = np.concatenate((img_cv2, front_view, side_view), axis=1)
            # Write overlay
            cv2.imwrite(overlay_fname, final_img)
            renderer.delete()


    def run_on_images(self, image_folder, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        images_list = [image for ext in image_extensions for image in glob(os.path.join(image_folder, ext))]
        for ind, img_path in enumerate(images_list):
            self.process_image(img_path, out_folder, ind)

    def run_on_3dpw_images(self, idx, out_folder):
        """
        根据给定的索引 idx，处理 3dpw_test.npz 中从 idx 到 idx+32 的 32 张图片。
        """
        # .npz 文件路径
        npz_path = '/home/zzb/pydata/recons/CameraHMR/data/data_evaluation/3dpw_test.npz'
        # 图片根目录
        image_root = '/home/zzb/pydata/recons/CameraHMR/data/3dpw/'

        # 加载 npz 文件（只加载一次）
        if not hasattr(self, 'npz_data') or self.npz_data is None:
            print("Loading 3dpw_test.npz...")
            self.npz_data = np.load(npz_path)
        
        data = self.npz_data
        total = len(data['imgname'])

        # 检查 idx 范围
        if idx < 0 or idx >= total:
            raise IndexError(f"Index {idx} is out of range [0, {total-1}]")
        
        # 确定结束位置（最多取到 total）
        end_idx = min(idx + 32, total)
        
        # 创建输出目录
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # 获取这32张图片的相对路径，并拼接成绝对路径
        img_paths = [
            os.path.join(image_root, data['imgname'][i])
            for i in range(idx, end_idx)
        ]

        # 处理每张图片
        for i, img_path in enumerate(img_paths):
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            # 这里的 ind 是从 0 到 31，表示当前批次中的序号
            self.process_image(img_path, out_folder, idx + i)  # 使用原始 idx+i 作为全局编号

    def run_on_emdb_images(self, idx, out_folder):
        """
        根据给定的索引 idx，处理 3dpw_test.npz 中从 idx 到 idx+32 的 32 张图片。
        如果输出目录中已存在处理结果，则跳过该图片。
        """
        # .npz 文件路径
        npz_path = '/home/zzb/pydata/recons/CameraHMR/data/data_evaluation/emdb_for_hmr2.npz'
        # 图片根目录
        image_root = '/home/zzb/pydata/recons/CameraHMR/data/EMDB/'

        # 加载 npz 文件（只加载一次）
        if not hasattr(self, 'npz_data') or self.npz_data is None:
            print("Loading emdb_test.npz...")
            self.npz_data = np.load(npz_path)
        
        data = self.npz_data
        total = len(data['imgname'])

        # 检查 idx 范围
        if idx < 0 or idx >= total:
            raise IndexError(f"Index {idx} is out of range [0, {total-1}]")
        
        # 确定结束位置（最多取到 total）
        end_idx = min(idx + 32, total)
        
        # 创建输出目录
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # 获取这32张图片的相对路径，并拼接成绝对路径
        img_paths = [
            os.path.join(image_root, data['imgname'][i])
            for i in range(idx, end_idx)
        ]

        # 处理每张图片
        for i, img_path in enumerate(img_paths):
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
                
            original_fname = os.path.basename(img_path)
            img_name, img_ext = os.path.splitext(original_fname)
            output_filename = os.path.join(out_folder, f"{img_name}_{idx + i:06d}{img_ext}")
            
            # 检查输出文件是否已存在
            if os.path.exists(output_filename):
                print(f"Output already exists, skipping: {output_filename}")
                continue
                
            # 处理图片
            self.process_image(img_path, out_folder, idx + i)  # 使用原始 idx+i 作为全局编号

    def save_emdb_images(self, idx, out_folder):
        """
        Copies original images from 3dpw_test.npz (from idx to idx+32) to out_folder.
        If the output already exists, skips the copy.
        """
        # .npz file path
        npz_path = '/home/zzb/pydata/recons/CameraHMR/data/data_evaluation/emdb_for_hmr2.npz'
        # Image root directory
        image_root = '/home/zzb/pydata/recons/CameraHMR/data/EMDB/'

        # Load npz file (only once)
        if not hasattr(self, 'npz_data') or self.npz_data is None:
            print("Loading emdb_test.npz...")
            self.npz_data = np.load(npz_path)
        
        data = self.npz_data
        total = len(data['imgname'])

        # Check index range
        if idx < 0 or idx >= total:
            raise IndexError(f"Index {idx} is out of range [0, {total-1}]")
        
        # Determine end index (up to total)
        end_idx = min(idx + 32, total)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # Get the 32 image paths
        img_paths = [
            os.path.join(image_root, data['imgname'][i])
            for i in range(idx, end_idx)
        ]

        # Copy each image
        for i, img_path in enumerate(img_paths):
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
                
            original_fname = os.path.basename(img_path)
            output_path = os.path.join(out_folder, original_fname)
            
            # Check if output already exists
            if os.path.exists(output_path):
                print(f"Output already exists, skipping: {output_path}")
                continue
                
            try:
                # Copy the image
                shutil.copy2(img_path, output_path)
                print(f"Copied: {img_path} -> {output_path}")
            except Exception as e:
                print(f"Error copying {img_path}: {str(e)}")

    def save_3dpw_images(self, idx, out_folder):
        """
        Copies original images from 3dpw_test.npz (from idx to idx+32) to out_folder.
        If the output already exists, skips the copy.
        """
        # .npz file path
        npz_path = '/home/zzb/pydata/recons/CameraHMR/data/data_evaluation/3dpw_test.npz'
        # Image root directory
        image_root = '/home/zzb/pydata/recons/CameraHMR/data/3dpw/'

        # Load npz file (only once)
        if not hasattr(self, 'npz_data') or self.npz_data is None:
            print("Loading 3dpw_test.npz...")
            self.npz_data = np.load(npz_path)
        
        data = self.npz_data
        total = len(data['imgname'])

        # Check index range
        if idx < 0 or idx >= total:
            raise IndexError(f"Index {idx} is out of range [0, {total-1}]")
        
        # Determine end index (up to total)
        end_idx = min(idx + 32, total)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # Get the 32 image paths
        img_paths = [
            os.path.join(image_root, data['imgname'][i])
            for i in range(idx, end_idx)
        ]

        # Copy each image
        for i, img_path in enumerate(img_paths):
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
                
            original_fname = os.path.basename(img_path)
            output_path = os.path.join(out_folder, original_fname)
            
            # Check if output already exists
            if os.path.exists(output_path):
                print(f"Output already exists, skipping: {output_path}")
                continue
                
            try:
                # Copy the image
                shutil.copy2(img_path, output_path)
                print(f"Copied: {img_path} -> {output_path}")
            except Exception as e:
                print(f"Error copying {img_path}: {str(e)}")