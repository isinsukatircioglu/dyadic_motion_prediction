import torch
import numpy as np


def process_bone(poses, start_joint, end_joint, old_val, goal_bone_len):
    bone_vec = poses[:, :, end_joint] - old_val
    new_old_val = poses[:, :, end_joint].clone()
    new_bone_vector = goal_bone_len * bone_vec / torch.norm(bone_vec, 2, dim=1).unsqueeze(1)
    poses[:, :, end_joint] = poses[:, :, start_joint] + new_bone_vector
    return poses, new_old_val


def traverse_body(poses, start_joint, old_val, bone_connections, visited_list, goal_bone_len):
    for count in range(len(bone_connections)):
        list_start_joint, list_end_joint = bone_connections[count]
        if start_joint == list_start_joint and not visited_list[count]:
            visited_list[count] = True
            poses, new_old_val = process_bone(poses, list_start_joint, list_end_joint, old_val, goal_bone_len[count])
            traverse_body(poses, list_end_joint, new_old_val, bone_connections, visited_list, goal_bone_len)
    return poses


def find_bone_lengths(poses, bone_connections):
    ##poses are of format (N,3,D)
    bone_len = torch.zeros([poses.shape[0], len(bone_connections)])
    bone_connections = np.array(bone_connections)
    i = bone_connections[:, 0]
    j = bone_connections[:, 1]
    bone_len = torch.norm(poses[:, :, i] - poses[:, :, j], 2, dim=1)
    assert bone_len.shape == (poses.shape[0], len(bone_connections))
    return bone_len


def convert_to_skeleton(poses, goal_bone_len, bone_connections):
    initial_poses = poses.clone()

    # make skeleton independent
    visited_list = [False] * len(bone_connections)

    # convert poses (this is a recursive function)
    converted_poses = traverse_body(initial_poses, bone_connections[0][0], initial_poses[:, :, bone_connections[0][0]],
                                    bone_connections, visited_list, goal_bone_len)
    return converted_poses


def centralize_normalize_rotate_poses(poses_union, joint_names, subject_1_pose, bone_connections, rotate=True, visualize=False):
    hip_index = joint_names.index('Hips')
    nb_joints = poses_union.shape[2]
    poses_union1 = poses_union[:, :, :nb_joints//2]
    poses_union2 = poses_union[:, :, nb_joints//2:]
    hip_pose1 = poses_union1[:, :, hip_index]
    hip_pose2 = poses_union2[:, :, hip_index]

    #translate the poses such that the hips of the first actor is at the center of the coord system
    poses1 = poses_union1 - hip_pose1.unsqueeze(2)
    poses2 = poses_union2 - hip_pose2.unsqueeze(2)
    normalized_poses = torch.cat((poses1, poses2), dim=-1)

    #first rotation: make everyone's shoulder vector [0, 1]
    shoulder_vector_1 = poses1[:, :, joint_names.index('LeftShoulder')] - poses1[:, :, joint_names.index('RightShoulder')]
    spine_vector_1 = poses1[:, :, joint_names.index('Neck')] - poses1[:, :, joint_names.index('Hips')]

    shoulder_vector_1 = shoulder_vector_1 / torch.norm(shoulder_vector_1, dim=1, keepdim=True)
    spine_vector_1 = spine_vector_1 / torch.norm(spine_vector_1, dim=1, keepdim=True)

    normal_vector_1 = torch.cross(shoulder_vector_1, spine_vector_1, dim=1)
    spine_vector_1 = torch.cross(normal_vector_1, shoulder_vector_1, dim=1)
    assert normal_vector_1.shape == shoulder_vector_1.shape

    inv_rotation_matrix = torch.inverse(torch.cat([shoulder_vector_1.unsqueeze(2), normal_vector_1.unsqueeze(2), spine_vector_1.unsqueeze(2)], dim=2))
    rotated_poses = torch.bmm(inv_rotation_matrix, poses1)

    #######################################

    shoulder_vector_2 = poses2[:, :, joint_names.index('LeftShoulder')] - poses2[:, :, joint_names.index('RightShoulder')]
    spine_vector_2 = poses2[:, :, joint_names.index('Neck')] - poses2[:, :, joint_names.index('Hips')]

    shoulder_vector_2 = shoulder_vector_2 / torch.norm(shoulder_vector_2, dim=1, keepdim=True)
    spine_vector_2 = spine_vector_2 / torch.norm(spine_vector_2, dim=1, keepdim=True)

    normal_vector_2 = torch.cross(shoulder_vector_2, spine_vector_2, dim=1)
    spine_vector_2 = torch.cross(normal_vector_2, shoulder_vector_2, dim=1)
    assert normal_vector_2.shape == shoulder_vector_2.shape

    inv_rotation_matrix_2 = torch.inverse(torch.cat([shoulder_vector_2.unsqueeze(2), normal_vector_2.unsqueeze(2), spine_vector_2.unsqueeze(2)], dim=2))
    rotated_poses_2 = torch.bmm(inv_rotation_matrix_2, poses2)


    return normalized_poses, inv_rotation_matrix, inv_rotation_matrix_2, rotated_poses, rotated_poses_2

