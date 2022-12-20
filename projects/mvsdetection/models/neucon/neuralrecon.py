import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict
from mmdet.models import DETECTORS 
from mmdet.models.builder import build_backbone, build_head, build_neck

@DETECTORS.register_module()
class NeuralRecon(nn.Module):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, pixel_mean, pixel_std, n_scales, loss_weights, backbone2d, head, fuse_to_global, train_cfg=None, test_cfg=None, pretrained=None):
        super(NeuralRecon, self).__init__()
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.n_scales = n_scales
        self.loss_weights = loss_weights
        self.backbone2d = build_backbone(backbone2d)
        self.neucon_net = build_head(head)
        self.fuse_to_global = build_neck(fuse_to_global)

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward_train(self, inputs, save_mesh=False):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        outputs = {}
        imgs = torch.unbind(inputs['imgs'], 1)

        # image feature extraction
        # in: images; out: feature maps
        features = [self.backbone2d(self.normalizer(img)) for img in imgs]

        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)

        return loss_dict

    def forward_test(self, inputs, save_mesh=False):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        outputs = {}
        imgs = torch.unbind(inputs['imgs'], 1)

        # image feature extraction
        # in: images; out: feature maps
        features = [self.backbone2d(self.normalizer(img)) for img in imgs]

        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)

        # fuse to global volume.
        if 'coords' in outputs.keys():
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)

        return [outputs]


    
    def parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['total_loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        data = self.data_converter(data)
        loss_dict = self(**data)
        loss, log_vars = self.parse_losses(loss_dict)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['imgs']))
        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        results = self(**data, return_loss=False)
        return results


    def forward(self, return_loss=True, rescale=False, save_mesh=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            inputs = kwargs
            return self.forward_train(inputs, save_mesh=save_mesh)
        else:
            inputs = self.data_converter(kwargs)
            return self.forward_test(inputs, save_mesh=save_mesh)
        
    def data_converter(self, data):
        data['world_to_aligned_camera'] = torch.stack(data['world_to_aligned_camera'], dim=0)
        data['proj_matrices'] = torch.stack(data['proj_matrices'], dim=0)
        data['vol_origin'] = torch.stack(data['vol_origin'], dim=0)
        data['vol_origin_partial'] = torch.stack(data['vol_origin_partial'], dim=0)
        if 'axis_align_matrix' in data:
            data['axis_align_matrix'] = torch.stack(data['axis_align_matrix'], dim=0)
        
        batch_size = len(data['tsdf_list'])
        n_scales = len(data['tsdf_list'][0])
        real_tsdf_list = []
        for i in range(n_scales):
            tsdf_list = []
            for j in range(batch_size):
                tsdf_list.append(data['tsdf_list'][j][i])
            tsdf_list = torch.stack(tsdf_list, dim=0)
            real_tsdf_list.append(tsdf_list)
        data['tsdf_list'] = real_tsdf_list
        
        real_occ_list = []
        for i in range(n_scales):
            occ_list = []
            for j in range(batch_size):
                occ_list.append(data['occ_list'][j][i])
            occ_list = torch.stack(occ_list, dim=0)
            real_occ_list.append(occ_list)
        data['occ_list'] = real_occ_list

        device = data['gt_labels_3d'][0].device 
        for i in range(len(data['gt_bboxes_3d'])):
            data['gt_bboxes_3d'][i] = data['gt_bboxes_3d'][i].to(device)

        return data 
    
    def init_weights(self):
        pass
