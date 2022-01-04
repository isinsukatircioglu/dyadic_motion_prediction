import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time
import h5py
import torch.optim as optim
from eval_arguments_multiple import Options
from attention_model_multiple import AttModel
from dataloader_lindyhop_multiple import Datasets_LindyHop
from utils import lr_decay_mine
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from visualize_3Dpose import show3Dpose
from log import save_csv_log, save_ckpt, save_options

def main(opt):

    print('>>> create models')
    in_features = opt.in_features  # 66
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model, num_stage=opt.num_stage,
                        dct_n=opt.dct_n)
    net_pred.cuda()
    model_path_len = '{}/ckpt_best.pth.tar'.format('./pretrained')

    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    print('>>> loading datasets')

    head = np.array(['act'])
    for k in range(1, opt.output_n + 1):
        head = np.append(head, [f'#{k}'])

    acts = ["p1", "p2"]
    errs = np.zeros([len(acts) + 1, opt.output_n])

    for i, act in enumerate(acts):
        test_dataset = Datasets_LindyHop(opt, split=2, actions=[act])
        print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
        test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True)

        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, act=i)
        print('testing error: {:.3f}'.format(ret_test['#1']))
        ret_log = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
        errs[i] = ret_log
    errs[-1] = np.mean(errs[:-1], axis=0)
    acts = np.expand_dims(np.array(acts + ["average"]), axis=1)
    value = np.concatenate([acts, errs.astype(np.str)], axis=1)
    save_csv_log(opt, head, value, is_create=True, file_name='ours_dyadic_test_samples_' + str(test_dataset.__len__()))


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, act=0):

    net_pred.eval()
    titles = np.array(range(opt.output_n)) + 1
    m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size
    joints_used = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    joint_nb = len(joints_used)
    # joints at same loc
    vis_poses = {}
    vis_poses['epoch'] = epo
    vis_poses['opt'] = opt
    itera = 1

    for i, (p3d_h36, p3d_h36_orig, rot_matrix, rot_matrix_aux) in enumerate(data_loader):
        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()
        p3d_h36_orig = p3d_h36_orig.float().cuda()
        rot_matrix = rot_matrix.float().cuda()
        p3d_src = p3d_h36.clone()
        p3d_h36 = p3d_h36[:, :, :3*joint_nb]
        p3d_sup = p3d_h36.clone()[:, -out_n - seq_in:].reshape([-1, seq_in + out_n, joint_nb, 3])
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera)
        # print('Output pred shape: ', p3d_out_all.shape)
        p3d_out = p3d_h36.clone()[:, in_n:in_n + opt.output_n]
        # p3d_out = p3d_out_all[:, seq_in:, 0]
        p3d_out = p3d_out_all[:, seq_in:]
        p3d_out_list = []
        for it in range(itera):
            p3d_out_list.append(p3d_out[:, :, it, :])
        p3d_out = torch.stack(p3d_out_list, dim=1).reshape([-1, out_n, joint_nb * 3])
        p3d_out = p3d_out.reshape([-1, out_n, joint_nb, 3])
        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, joint_nb, 3])
        p3d_out_all = p3d_out_all.reshape([batch_size, seq_in + out_n, itera, joint_nb, 3])
        p3d_out_all_future = p3d_out_all[:, seq_in:, :, :, :]
        p3d_out_all_fst_past = p3d_out_all[:, :seq_in, 0, :, :]
        p3d_out_all = torch.cat((p3d_out_all_fst_past, p3d_out_all_future.reshape(batch_size, out_n, joint_nb, 3)), dim=1)

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()


    ret = {}
    m_p3d_h36 = m_p3d_h36 / n
    print('Test avg error', np.array(m_p3d_h36).sum() / out_n)
    for j in range(out_n):
        ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    print(ret)
    return ret

if __name__ == '__main__':
    option = Options().parse()
    main(option)
