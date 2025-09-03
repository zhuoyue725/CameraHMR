import numpy as np
from loguru import logger
from utils.eval_utils import compute_similarity_transform_batch

# 假设你已经有了 SMPL 模型的接口，例如来自 pytorch3d、spin、hmr 等
# 这里我们假设使用的是 SPIN 中的 SMPL 模型（来自 smplx 库）
from smplx import SMPL
import torch
import cv2

# 全局加载 SMPL 模型（避免重复加载）
smpl = SMPL(model_path='data/models/SMPL', gender='neutral', batch_size=1).cuda()

def compute_pampjpe(pred_poses, pred_betas, gt_poses, gt_betas, has_conf=False, reduction='mean'):
    """
    Compute PA-MPJPE between predicted and ground truth SMPL parameters.

    Args:
        pred_poses: (23, 3, 3) numpy array, predicted pose rotation matrices (excluding root)
        pred_betas: (10,) numpy array, predicted shape parameters
        gt_poses: (23, 3, 3) numpy array, ground truth pose rotation matrices
        gt_betas: (10,) numpy array, ground truth shape parameters
        has_conf: bool, whether to use confidence weighting (not used here)
        reduction: 'mean' or 'sum' or 'none'

    Returns:
        pampjpe: scalar (float), PA-MPJPE in mm
        pred_joints: (N, 24, 3) aligned predicted joints
        gt_joints: (N, 24, 3) ground truth joints
    """
    global smpl

    # 1. 扩展为 batch 维度 (N=1)
    pred_poses = torch.tensor(pred_poses, dtype=torch.float32).unsqueeze(0)  # (1, 23, 3, 3)
    pred_betas = torch.tensor(pred_betas, dtype=torch.float32).unsqueeze(0)  # (1, 10)
    gt_poses = torch.tensor(gt_poses, dtype=torch.float32).unsqueeze(0)      # (1, 23, 3, 3)
    gt_betas = torch.tensor(gt_betas, dtype=torch.float32).unsqueeze(0)      # (1, 10)

    device = pred_poses.device

    # 2. 构造完整 24 个关节的旋转矩阵（添加 root joint）
    # pred: root + 23
    root_rot = torch.eye(3).expand(1, 1, 3, 3)  # 假设根节点旋转为单位阵（或你有 pred_root?）
    pred_full_pose = torch.cat([root_rot, pred_poses], dim=1).cuda()  # (1, 24, 3, 3)
    gt_full_pose = torch.cat([root_rot, gt_poses], dim=1).cuda()      # (1, 24, 3, 3)

    # 转为 axis-angle (SMPL 需要 24*3 的向量)
    def rotmat_to_axis_angle(R):
        # R: (B, 24, 3, 3)
        B = R.shape[0]
        R = R.reshape(-1, 3, 3)
        aa = []
        for r in R:
            vec, _ = cv2.Rodrigues(r.cpu().numpy())
            aa.append(vec)
        aa = np.array(aa).reshape(B, 24, 3)
        return torch.tensor(aa, dtype=torch.float32).cuda()

    pred_aa = rotmat_to_axis_angle(pred_full_pose).view(1, -1)  # (1, 72)
    gt_aa = rotmat_to_axis_angle(gt_full_pose).view(1, -1)      # (1, 72)

    # 3. 通过 SMPL 模型前向传播得到 3D 关键点
    with torch.no_grad():
        # Predicted
        pred_output = smpl(
            betas=pred_betas.cuda(),
            body_pose=pred_aa[:, 3:],  # exclude root
            global_orient=pred_aa[:, :3],
            pose2rot=False  # 因为我们输入的是 axis-angle
        )
        pred_joints = pred_output.joints[:, :24, :].cpu().numpy()  # (1, 24, 3)

        # Ground Truth
        gt_output = smpl(
            betas=gt_betas.cuda(),
            body_pose=gt_aa[:, 3:],
            global_orient=gt_aa[:, :3],
            pose2rot=False
        )
        gt_joints = gt_output.joints[:, :24, :].cpu().numpy()  # (1, 24, 3)

    # 4. 计算 PA-MPJPE (Procrustes alignment)
    pa_pred_joints = compute_similarity_transform_batch(pred_joints, gt_joints)  # (1, 24, 3)

    # 5. 计算误差 (mm)
    per_joint_error = np.linalg.norm(pa_pred_joints - gt_joints, axis=-1)  # (1, 24)
    pampjpe = per_joint_error.mean() if reduction == 'mean' else per_joint_error.sum()

    return pampjpe, pa_pred_joints, gt_joints



# 示例输入
pred_poses = np.random.randn(23, 3, 3)  # (23, 3, 3)
pred_betas = np.random.randn(10,)       # (10,)
gt_poses = np.random.randn(23, 3, 3)
gt_betas = np.random.randn(10,)

# 确保是旋转矩阵（正交，det=1），这里仅为示例

pampjpe, aligned_pred, gt = compute_pampjpe(pred_poses, pred_betas, gt_poses, gt_betas)

print(f"PA-MPJPE: {pampjpe:.2f} mm")