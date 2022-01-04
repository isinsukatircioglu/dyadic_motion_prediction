######DATASET LINDYHOP######
from torch.utils.data import Dataset
import scipy.io as sio
from matplotlib import pyplot as plt
import json
import numpy as np
import torch
from normalization_multiple import centralize_normalize_rotate_poses


class Datasets_LindyHop(Dataset):
    def __init__(self, opt, actions=None, split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
        self.path_to_data = "./data"
        self.split = split
        self.sample_rate = 2
        self.in_n = opt.input_n
        self.out_n = opt.output_n
        self.p3d = {}
        self.p3d_orig = {}
        self.rot_poses = {}
        self.rot_poses_aux = {}
        self.p3d_rot_matrix = {}
        self.p3d_rot_matrix_aux = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        subs = np.array([[4], [4], [4]])
        #subs = np.array([[1, 2, 5, 6, 7, 8, 9], [3], [4]]) # Once the full dataset is released, uncomment this line
        acts_default = ['p1', 'p2']

        if actions is None:
            acts = ['p1', 'p2']
        else:
            acts = actions
        self.joint_names = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightAnkle", "RightToeBase", "LeftUpLeg",
                            "LeftLeg",
                            "LeftFoot",
                            "LeftAnkle", "LeftToeBase", "Neck", "Head", "LeftShoulder", "LeftForeArm",
                            "LeftHand", "RightShoulder", "RightForeArm", "RightHand"]
        dim_used = [8, 9, 10, 11, 22, 23, 12, 13, 14, 19, 20, 1, 0, 5, 6, 7, 2, 3, 4]

        I = np.array([0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 11, 13, 14, 11, 16, 17])
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

        bone_connections = [[0, 11], [11, 12], [11, 16], [16, 17], [17, 18], [11, 13], [13, 14], [14, 15], [0, 1],
                            [1, 2],
                            [2, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [8, 9], [9, 10]]

        # Read subject 1 pose from training
        subj = subs[0][0]
        filename = '{0}/seq{1}_poses.json'.format(self.path_to_data, subj)
        f = open(filename, "r")
        print('choose the anchor pose from his file: ', filename)
        data = json.loads(f.read())
        p1 = data['p1']
        f.close()
        p1 = np.array(p1)
        p1 = torch.from_numpy(p1).float().cuda()
        p1 = p1[:, dim_used, :]
        p1 = p1[:, :, [0, 2, 1]]
        first_pose = p1.permute(0, 2, 1)[0]

        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                print("Reading subject {0}, action {1}".format(subj, action))
                filename = '{0}/seq{1}_poses.json'.format(self.path_to_data, subj)
                f = open(filename, "r")
                data = json.loads(f.read())
                p1 = data[action]
                p1 = np.array(p1)
                print(filename)
                print('seq before sampling', p1.shape)
                p1 = torch.from_numpy(p1).float().cuda()
                # read the interacting people's data
                action_default_index = acts_default.index(action)
                pairing_index = abs(action_default_index - 1)
                p2 = data[acts_default[pairing_index]]
                p2 = np.array(p2)
                p2 = torch.from_numpy(p2).float().cuda()
                f.close()
                p1 = p1[:, dim_used, :]
                p2 = p2[:, dim_used, :]
                p1 = p1[:, :, [0, 2, 1]]
                p2 = p2[:, :, [0, 2, 1]]
                the_sequence = torch.cat((p1, p2), dim=1)
                even_list = range(0, the_sequence.shape[0], self.sample_rate)
                the_sequence = the_sequence[even_list, :, :]
                print('seq after sampling', the_sequence.shape)
                # put the interacting people together
                num_frames = the_sequence.shape[0]
                # pelvis subtraction - skeleton norm - rotation
                the_sequence_orig = the_sequence.clone()
                the_sequence, rot_matrix, rot_matrix_aux, rotated_poses, rotated_poses_aux = centralize_normalize_rotate_poses(the_sequence.permute(0, 2, 1), self.joint_names, first_pose, bone_connections)
                the_sequence = the_sequence.permute(0, 2, 1)
                rotated_poses = rotated_poses.permute(0, 2, 1)[:, :len(dim_used), :]
                rotated_poses_aux = rotated_poses_aux.permute(0, 2, 1)[:, :len(dim_used), :]
                print('the_sequence shape', the_sequence.shape)
                self.p3d[key] = the_sequence.contiguous().view(num_frames, -1).cpu().data.numpy()
                self.p3d_orig[key] = the_sequence_orig.contiguous().view(num_frames, -1).cpu().data.numpy()
                self.p3d_rot_matrix[key] = rot_matrix.cpu().data.numpy()
                self.p3d_rot_matrix_aux[key] = rot_matrix_aux.cpu().data.numpy()
                self.rot_poses[key] =  rotated_poses.contiguous().view(num_frames, -1).cpu().data.numpy()
                self.rot_poses_aux[key] =  rotated_poses_aux.contiguous().view(num_frames, -1).cpu().data.numpy()
                valid_frames = np.arange(0, num_frames - seq_len + 1, opt.skip_rate)
                tmp_data_idx_1 = [key] * len(valid_frames)
                tmp_data_idx_2 = list(valid_frames)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                key += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        ################
        #Rotate the poses in the sequence wrt the reference pose
        n_joints = self.p3d[key][fs].shape[1] // 3 // 2
        poses_primary = self.p3d[key][fs].reshape(self.in_n + self.out_n, self.p3d[key][fs].shape[1] // 3, 3)[:, :n_joints, :]
        poses_aux = self.p3d[key][fs].reshape(self.in_n + self.out_n, self.p3d[key][fs].shape[1] // 3, 3)[:, n_joints:, :]
        p3d_transformed_primary = (np.repeat(self.p3d_rot_matrix[key][start_frame][np.newaxis, :, :], self.in_n + self.out_n, axis=0, ) @ poses_primary.transpose(0, 2, 1)).transpose(0, 2, 1)
        p3d_transformed_aux = (np.repeat(self.p3d_rot_matrix_aux[key][start_frame][np.newaxis, :, :], self.in_n + self.out_n, axis=0, ) @ poses_aux.transpose(0, 2, 1)).transpose(0, 2, 1)
        p3d_transformed = np.concatenate((p3d_transformed_primary, p3d_transformed_aux), axis=-2).reshape(self.in_n + self.out_n, self.p3d[key][fs].shape[1])
        #################
        return p3d_transformed, self.p3d_orig[key][fs], np.linalg.inv(self.p3d_rot_matrix[key][start_frame])[np.newaxis, :, :], np.linalg.inv(self.p3d_rot_matrix_aux[key][start_frame])[np.newaxis, :, :]

