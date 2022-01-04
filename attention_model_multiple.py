import torch
from torch.nn import Module
from torch import nn
import numpy as np
import math
from gcn import GCN
from data_utils import get_dct_matrix

from torch.nn import Module
from torch import nn
import math


class AttModel(Module):
    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())


        self.gcn = GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                       num_stage=num_stage,
                       node_n=in_features)

        self.merge_pairwise = nn.Conv1d(in_channels=(dct_n) * 2, out_channels=(dct_n), kernel_size=1,
                              bias=False)

        self.merge = nn.Conv1d(in_channels=(dct_n) * 2, out_channels=(dct_n), kernel_size=1,
                               bias=False)

    def forward(self, src, output_n=25, input_n=50, itera=1, dim_one=19*3):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp_primary = src.clone()[:, :, :dim_one]
        src_tmp_aux = src.clone()[:, :, dim_one:]
        #Comment out for actual aux pose
        src_tmp_aux = src_tmp_primary - src_tmp_aux # Relative motion
        bs = src.shape[0]

        src_key_tmp_primary = src_tmp_primary.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp_primary = src_tmp_primary.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        src_key_tmp_aux = src_tmp_aux.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp_aux = src_tmp_aux.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp_primary = src_tmp_primary[:, idx].clone().reshape([bs * vn, vl, -1])
        src_value_tmp_primary = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp_primary).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11]

        src_value_tmp_aux = src_tmp_aux[:, idx].clone().reshape([bs * vn, vl, -1])
        src_value_tmp_aux = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp_aux).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11]

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp_primary = self.convK(src_key_tmp_primary / 1000.0)
        key_tmp_aux = self.convK(src_key_tmp_aux / 1000.0)

        query_tmp_primary = self.convQ(src_query_tmp_primary / 1000.0) #q1
        query_tmp_aux = self.convQ(src_query_tmp_aux / 1000.0) #q2

        # self-attention for primary pose
        score_tmp_primary = torch.matmul(query_tmp_primary.transpose(1, 2), key_tmp_primary) + 1e-15
        att_tmp_primary = score_tmp_primary / (torch.sum(score_tmp_primary, dim=2, keepdim=True))
        dct_att_tmp_primary = torch.matmul(att_tmp_primary, src_value_tmp_primary)[:, 0].reshape([bs, -1, dct_n]) #U1

        #pairwise attention query primary
        score_tmp_21 = torch.matmul(query_tmp_primary.transpose(1, 2), key_tmp_aux) + 1e-15
        att_tmp_21 = score_tmp_21 / (torch.sum(score_tmp_21, dim=2, keepdim=True))
        dct_att_tmp_21 = torch.matmul(att_tmp_21, src_value_tmp_aux)[:, 0].reshape([bs, -1, dct_n])  # U21

        #pairwise attention query aux
        score_tmp_12 = torch.matmul(query_tmp_aux.transpose(1, 2), key_tmp_primary) + 1e-15
        att_tmp_12 = score_tmp_12 / (torch.sum(score_tmp_12, dim=2, keepdim=True))
        dct_att_tmp_12 = torch.matmul(att_tmp_12, src_value_tmp_primary)[:, 0].reshape([bs, -1, dct_n])  # U12

        #P1 computation
        dct_att_tmp_pairwise = self.merge_pairwise(torch.cat((dct_att_tmp_21, dct_att_tmp_12), dim=-1).permute(0, 2, 1)).permute(0, 2, 1)


        #GCN for primary (last observed seq D1 from the primary subject)
        input_gcn_primary = src_tmp_primary[:, idx]
        dct_in_tmp_primary = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn_primary).transpose(1, 2)
        dct_in_tmp_primary = torch.cat([dct_in_tmp_primary, dct_att_tmp_primary], dim=-1)
        dct_out_tmp_primary = self.gcn(dct_in_tmp_primary)
        out_gcn_primary = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0), dct_out_tmp_primary[:, :, :dct_n].transpose(1, 2))

        # GCN for pairwise (last observed seq D1 from the primary subject)
        input_gcn_pairwise = src_tmp_primary[:, idx]
        dct_in_tmp_pairwise = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn_pairwise).transpose(1, 2)
        dct_in_tmp_pairwise = torch.cat([dct_in_tmp_pairwise, dct_att_tmp_pairwise], dim=-1)
        dct_out_tmp_pairwise = self.gcn(dct_in_tmp_pairwise)
        out_gcn_pairwise = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0), dct_out_tmp_pairwise[:, :, :dct_n].transpose(1, 2))

        #MERGE GCN OUTPUTS
        out_gcn = self.merge(torch.cat((out_gcn_primary, out_gcn_pairwise), dim=-2))
        out_gcn = out_gcn + out_gcn_primary # residual

        outputs.append(out_gcn.unsqueeze(2))
        outputs = torch.cat(outputs, dim=2)

        return outputs
