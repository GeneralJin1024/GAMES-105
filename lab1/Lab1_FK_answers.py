from math import degrees
from ntpath import join
from unittest import skip
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    with open(bvh_file_path, 'r') as f:
        root_idx = -1
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if line.startswith('Frame Time'):
                break
            if '{' in line or '}' in line:
                if '{' in line:
                    root_idx = len(joint_parent) - 1
                if '}' in line:
                    root_idx = joint_parent[root_idx]
                continue
            tokens = line.split()
            if tokens[0] in ('ROOT', 'JOINT'):
                joint_name.append(tokens[1])
                joint_parent.append(root_idx)
            if tokens[0] == 'OFFSET':
                data = [float(tokens[i]) for i in range(1,4)]
                joint_offset.append(np.array(data).reshape(1, -1))  # 1行，x列
            if line.startswith('End Site'):
                joint_name.append(joint_name[root_idx] + '_end')
                joint_parent.append(root_idx)
    joint_offset = np.concatenate(joint_offset, axis=0)
    #print(joint_name)
    #print(joint_parent)
    #print(joint_offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []
    frame_data = motion_data[frame_id]
    for idx, pidx in enumerate(joint_parent):
        offset = joint_offset[idx]
        if pidx == -1:
            joint_positions.append(np.array(frame_data[:3]).reshape(1, -1)+offset)
            frame_data = frame_data[3:]
            joint_orientations.append(R.from_euler('XYZ', frame_data[:3], degrees=True).as_quat())
            frame_data = frame_data[3:]
        else:
            name = joint_name[idx]
            parent_position = joint_positions[pidx]
            if '_end' in name:
                parent_rotation = joint_orientations[pidx]
                local_position = parent_rotation * np.concatenate(([0], offset)) * parent_rotation.conj()
                local_position = local_position[1:]
                joint_orientations.append(R.from_euler('XYZ', [0,0,0], degrees=True).as_quat())
            else:
                parent_rotation_m = R.from_quat(joint_orientations[pidx]).as_matrix()
                rotation_m = parent_rotation_m.dot(R.from_euler('XYZ', frame_data[:3], degrees=True).as_matrix())
                frame_data = frame_data[3:]
                joint_orientations.append(R.from_matrix(rotation_m).as_quat())
                local_position = parent_position+parent_rotation_m.dot(offset)
            joint_positions.append(np.array(local_position).reshape(1, -1))
    joint_positions = np.concatenate(joint_positions, axis=0)
    joint_orientations = [np.asarray(r).reshape(1,-1) for r in joint_orientations]
    joint_orientations = np.concatenate(joint_orientations, axis=0)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    a_pose_joint_name, _, _ = part1_calculate_T_pose(A_pose_bvh_path)
    t_pose_joint_name, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    a_motion_data = load_motion_data(A_pose_bvh_path)
    a_skip_data_cnt = []
    skip_cnt = 0
    # 跳过最末尾骨骼，因为末尾骨骼没有旋转
    for name in a_pose_joint_name:
        if name.endswith('_end'):
            skip_cnt += 1
        a_skip_data_cnt.append(skip_cnt)
    motion_data = []
    for frame_id in range(a_motion_data.shape[0]):
        single_frame_motion_data = []
        for joint_name in t_pose_joint_name:
            A_pose_index=a_pose_joint_name.index(joint_name)
            if joint_name == 'RootJoint':
                single_frame_motion_data.append(a_motion_data[frame_id, :3])
                single_frame_motion_data.append(a_motion_data[frame_id, 3:6])
            elif joint_name == 'rShoulder':
                ori_rot = R.from_euler('XYZ', [0,0,45], degrees=True).as_matrix()
                local_rot = R.from_euler('XYZ', a_motion_data[frame_id, (A_pose_index-a_skip_data_cnt[A_pose_index]+1)*3:(A_pose_index-a_skip_data_cnt[A_pose_index]+1)*3+3], degrees=True).as_matrix()
                rot = local_rot.dot(ori_rot)
                single_frame_motion_data.append(R.from_matrix(rot).as_euler('XYZ', degrees=True))
            elif joint_name == 'lShoulder':
                ori_rot = R.from_euler('XYZ', [0,0,-45], degrees=True).as_matrix()
                local_rot = R.from_euler('XYZ', a_motion_data[frame_id, (A_pose_index-a_skip_data_cnt[A_pose_index]+1)*3:(A_pose_index-a_skip_data_cnt[A_pose_index]+1)*3+3], degrees=True).as_matrix()
                rot = local_rot.dot(ori_rot)
                single_frame_motion_data.append(R.from_matrix(rot).as_euler('XYZ', degrees=True))
            elif joint_name.endswith('_end'):
                continue
            else:
                single_frame_motion_data.append(a_motion_data[frame_id, (A_pose_index-a_skip_data_cnt[A_pose_index]+1)*3:(A_pose_index-a_skip_data_cnt[A_pose_index]+1)*3+3])
        single_frame_motion_data = np.concatenate(single_frame_motion_data, axis=0)
        motion_data.append(single_frame_motion_data)
    return np.array(motion_data)
