"""
相机位姿 SLERP (Spherical Linear Interpolation) 插值模块。

实现两个相机位姿之间的球面线性插值：
- 旋转部分使用四元数 SLERP（保证旋转的平滑性）
- 平移部分使用线性插值 (LERP)
"""

import numpy as np
from typing import List, Tuple
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)



def interpolate_camera_poses(
    cam_params: List[Camera],
    src_indices: np.ndarray,
    tgt_indices: np.ndarray,
) -> List[Camera]:
    """
    基于所有原始帧建立全局插值函数，在 tgt_indices 处采样得到插值后的位姿。

    旋转部分使用 scipy Slerp（多关键帧球面线性插值），
    平移部分使用 scipy interp1d 线性插值。。

    参数:
        cam_params: 原始帧的 Camera 对象列表，长度与 src_indices 一致。
        src_indices: 原始帧索引数组（浮点或整数），长度与 cam_params 一致。
        tgt_indices: 目标 latent 帧索引数组（浮点），即需要插值得到位姿的位置。

    返回:
        list[Camera]，长度为 len(tgt_indices) 的插值后相机序列。
    """
    src_indices = np.asarray(src_indices, dtype=np.float64)
    tgt_indices = np.asarray(tgt_indices, dtype=np.float64)

    # 提取所有帧的 w2c 旋转矩阵和平移向量
    src_rot_mat = np.array([cam.w2c_mat[:3, :3] for cam in cam_params])  # [N, 3, 3]
    src_trans_vec = np.array([cam.w2c_mat[:3, 3] for cam in cam_params])  # [N, 3]

    # 处理左手坐标系：检测行列式符号，必要时翻转 Z 轴
    dets = np.linalg.det(src_rot_mat)
    flip_handedness = dets.size > 0 and np.median(dets) < 0.0
    if flip_handedness:
        flip_mat = np.diag([1.0, 1.0, -1.0]).astype(src_rot_mat.dtype)
        src_rot_mat = src_rot_mat @ flip_mat

    # 平移：线性插值
    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind='linear',
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    # 旋转：全局 Slerp 插值
    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()  # [N, 4] (x, y, z, w)
    # 确保相邻四元数无符号突变
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    if flip_handedness:
        interpolated_rot_mat = interpolated_rot_mat @ flip_mat

    ref_cam = cam_params[0]
    result_cameras = []
    for i in range(len(tgt_indices)):
        w2c_3x4 = np.hstack([interpolated_rot_mat[i], interpolated_trans_vec[i].reshape(3, 1)])
        entry = np.zeros(19, dtype=np.float32)
        entry[1:5] = [ref_cam.fx, ref_cam.fy, ref_cam.cx, ref_cam.cy]
        entry[7:] = w2c_3x4.reshape(12)
        result_cameras.append(Camera(entry))

    return result_cameras


