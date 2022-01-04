import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time
import h5py
import torch.optim as optim
from read_arguments_multiple import Options
from attention_model_multiple import AttModel
from dataloader_lindyhop_multiple import Datasets_LindyHop
from utils import lr_decay_mine
import numpy as np
from log import save_csv_log, save_ckpt, save_options

def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 66
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model, num_stage=opt.num_stage,
                        dct_n=opt.dct_n)
    net_pred.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    if opt.is_load or opt.is_eval:
        model_path_len = './pretrained/ckpt_best.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        #start_epoch = ckpt['epoch'] + 1
        start_epoch =  1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt len loaded (err: {})".format(ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        dataset = Datasets_LindyHop(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_dataset = Datasets_LindyHop(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

    test_dataset = Datasets_LindyHop(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            save_ckpt({'epoch': epo,
                       'lr': lr_now,
                       'err': ret_valid['m_p3d_h36'],
                       'state_dict': net_pred.state_dict(),
                       'optimizer': optimizer.state_dict()},
                      is_best=is_best, opt=opt)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size
    joints_used = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    joint_nb = len(joints_used)
    itera = opt.itera
    st = time.time()
    for i, (p3d_h36, p3d_h36_orig, rot_matrix, rot_matrix_aux) in enumerate(data_loader):
        # print(i)
        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()
        p3d_src = p3d_h36.clone()
        p3d_h36 = p3d_h36[:, :, :3*joint_nb]
        p3d_sup = p3d_h36.clone()[:, -out_n - seq_in:].reshape([-1, seq_in + out_n, joint_nb, 3])
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera)
        p3d_out = p3d_h36.clone()[:, in_n:in_n + opt.output_n]
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
        p3d_out_all = torch.cat((p3d_out_all_fst_past, p3d_out_all_future.reshape(batch_size, out_n, joint_nb, 3)),
                                dim=1)
        grad_norm = 0
        if is_train == 0:
            loss_p3d = torch.mean(torch.norm(p3d_out_all - p3d_sup, dim=3))
            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))


    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        print('Test avg error', np.array(m_p3d_h36).sum() / out_n)
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
        print(ret)
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
