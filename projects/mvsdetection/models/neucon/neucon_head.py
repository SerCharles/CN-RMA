import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor
from loguru import logger
from mmdet.models import HEADS 
from mmdet.models.builder import build_neck


from projects.mvsdetection.ops.back_project import back_project
from projects.mvsdetection.ops.generate_grids import generate_grid

@HEADS.register_module
class NeuConHead(nn.Module):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, alpha, n_layers, voxel_size, n_voxels, sparsereg_dropout, pos_weight, num_sample, thresholds, use_gru_fusion):
        super(NeuConHead, self).__init__()
        self.alpha = alpha
        self.n_layers = n_layers
        self.voxel_size = voxel_size
        self.n_voxels = n_voxels 
        self.sparsereg_dropout = sparsereg_dropout
        self.pos_weight = pos_weight
        self.num_sample = num_sample
        self.thresholds = thresholds 
        self.n_scales = len(self.thresholds) - 1
        self.use_gru_fusion = use_gru_fusion

        channels_feature = [80, 40, 24]
        channels = [96, 48, 24]

        ch_in = [channels_feature[0] + 1, channels[0] + channels_feature[1] + 2 + 1, 
                 channels[1] + channels_feature[2] + 2 + 1, channels[2] + channels_feature[2] + 2 + 1]

        gru_fusion_dict = dict(type='GRUFusion',
                               direct_substitute=False,
                               ch_in=channels,
                               thresholds=self.thresholds,
                               n_layers=self.n_layers,
                               voxel_size=self.voxel_size,
                               n_voxels=self.n_voxels,
                               use_gru_fusion=self.use_gru_fusion)
        self.gru_fusion = build_neck(gru_fusion_dict)
        # sparse conv
        self.sp_convs = nn.ModuleList()
        # MLPs that predict tsdf and occupancy.
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()
        for i in range(len(self.thresholds)):
            spvcnn_dict = dict(type='SPVCNN',
                               num_classes=1,
                               in_channels=ch_in[i],
                               pres=1,
                               cr=1 / 2 ** i,
                               vres=self.voxel_size * 2 ** (self.n_scales - i),
                               dropout=self.sparsereg_dropout)
            spvcnn = build_neck(spvcnn_dict)
            self.sp_convs.append(spvcnn)
            self.tsdf_preds.append(nn.Linear(channels[i], 1))
            self.occ_preds.append(nn.Linear(channels[i], 1))

    def get_target(self, coords, inputs, scale):
        '''
        Won't be used when 'fusion_on' flag is turned on
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param inputs: (List), inputs['tsdf_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
        :param scale:
        :return: tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
        :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
        '''
        with torch.no_grad():
            tsdf_target = inputs['tsdf_list'][scale]
            occ_target = inputs['occ_list'][scale]
            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return tsdf_target, occ_target

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        '''

        :param pre_feat: (Tensor), features from last level, (N, C)
        :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        :param interval: interval of voxels, interval = scale ** 2
        :param num: 1 -> 8
        :return: up_feat : (Tensor), upsampled features, (N*8, C)
        :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        '''
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def forward(self, features, inputs, outputs):
        '''

        :param features: list: features for each image: eg. list[0] : pyramid features for image0 : [(B, C0, H, W), (B, C1, H/2, W/2), (B, C2, H/2, W/2)]
        :param inputs: meta data from dataloader
        :param outputs: {}
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
        }
        :return: loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
        }
        '''
        bs = features[0][0].shape[0]
        pre_feat = None
        pre_coords = None
        loss_dict = {}
        for i in range(self.n_layers):
            loss = torch.Tensor(np.array([0]))[0].cuda()
            loss_dict.update({f'tsdf_occ_loss_{i}': loss})
            
        # ----coarse to fine----
        for i in range(self.n_layers):
            interval = 2 ** (self.n_scales - i)
            scale = self.n_scales - i

            if i == 0:
                # ----generate new coords----
                coords = generate_grid(self.n_voxels, interval)[0]
                up_coords = []
                for b in range(bs):
                    up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
            else:
                # ----upsample coords----
                up_feat, up_coords = self.upsample(pre_feat, pre_coords, interval)

            # ----back project----
            feats = torch.stack([feat[scale] for feat in features])
            KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()
            volume, count = back_project(up_coords, inputs['vol_origin_partial'], self.voxel_size, feats, KRcam)
            grid_mask = count > 1

            # ----concat feature from last stage----
            if i != 0:
                feat = torch.cat([volume, up_feat], dim=1)
            else:
                feat = volume

            if not self.use_gru_fusion:
                tsdf_target, occ_target = self.get_target(up_coords, inputs, scale)

            # ----convert to aligned camera coordinate----
            r_coords = up_coords.detach().clone().float()
            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == b).squeeze(1)
                coords_batch = up_coords[batch_ind][:, 1:].float()
                coords_batch = coords_batch * self.voxel_size + inputs['vol_origin_partial'][b].float()
                coords_batch = torch.cat((coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1)
                coords_batch = coords_batch @ inputs['world_to_aligned_camera'][b, :3, :].permute(1, 0).contiguous()
                r_coords[batch_ind, 1:] = coords_batch

            # batch index is in the last position
            r_coords = r_coords[:, [1, 2, 3, 0]]

            # ----sparse conv 3d backbone----
            point_feat = PointTensor(feat, r_coords)
            feat = self.sp_convs[i](point_feat)

            # ----gru fusion----
            if self.use_gru_fusion:
                up_coords, feat, tsdf_target, occ_target = self.gru_fusion(up_coords, feat, inputs, i)
                grid_mask = torch.ones_like(feat[:, 0]).bool()

            tsdf = self.tsdf_preds[i](feat)
            occ = self.occ_preds[i](feat)

            # -------compute loss-------
            if tsdf_target is not None:
                loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target,
                                         mask=grid_mask,
                                         pos_weight=self.pos_weight)
            else:
                loss = torch.Tensor(np.array([0]))[0]
            loss_dict.update({f'tsdf_occ_loss_{i}': loss})

            # ------define the sparsity for the next stage-----
            occupancy = occ.squeeze(1) > self.thresholds[i]
            occupancy[grid_mask == False] = False

            num = int(occupancy.sum().data.cpu())

            if num == 0:
                logger.warning('no valid points: scale {}'.format(i))
                return outputs, loss_dict

            # ------avoid out of memory: sample points if num of points is too large-----
            if self.training and num > self.num_sample[i] * bs:
                choice = np.random.choice(num, num - self.num_sample[i] * bs,
                                          replace=False)
                ind = torch.nonzero(occupancy)
                occupancy[ind[choice]] = False

            pre_coords = up_coords[occupancy]
            for b in range(bs):
                batch_ind = torch.nonzero(pre_coords[:, 0] == b).squeeze(1)
                if len(batch_ind) == 0:
                    logger.warning('no valid points: scale {}, batch {}'.format(i, b))
                    return outputs, loss_dict

            pre_feat = feat[occupancy]
            pre_tsdf = tsdf[occupancy]
            pre_occ = occ[occupancy]

            pre_feat = torch.cat([pre_feat, pre_tsdf, pre_occ], dim=1)

            if i == self.n_layers - 1:
                outputs['coords'] = pre_coords
                outputs['tsdf'] = pre_tsdf

        return outputs, loss_dict

    @staticmethod
    def compute_loss(tsdf, occ, tsdf_target, occ_target, loss_weight=(1, 1),
                     mask=None, pos_weight=1.0):
        '''

        :param tsdf: (Tensor), predicted tsdf, (N, 1)
        :param occ: (Tensor), predicted occupancy, (N, 1)
        :param tsdf_target: (Tensor),ground truth tsdf, (N, 1)
        :param occ_target: (Tensor), ground truth occupancy, (N, 1)
        :param loss_weight: (Tuple)
        :param mask: (Tensor), mask voxels which cannot be seen by all views
        :param pos_weight: (float)
        :return: loss: (Tensor)
        '''
        # compute occupancy/tsdf loss
        tsdf = tsdf.view(-1)
        occ = occ.view(-1)
        tsdf_target = tsdf_target.view(-1)
        occ_target = occ_target.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            tsdf = tsdf[mask]
            occ = occ[mask]
            tsdf_target = tsdf_target[mask]
            occ_target = occ_target[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            logger.warning('target: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0] * tsdf.sum()
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight

        # compute occ bce loss
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)

        # compute tsdf l1 loss
        tsdf = apply_log_transform(tsdf[occ_target])
        tsdf_target = apply_log_transform(tsdf_target[occ_target])
        tsdf_loss = torch.mean(torch.abs(tsdf - tsdf_target))

        # compute final loss
        loss = loss_weight[0] * occ_loss + loss_weight[1] * tsdf_loss
        return loss


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out