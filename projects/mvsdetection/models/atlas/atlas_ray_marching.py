import numpy as np
import os
from math import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from mmdet.models import DETECTORS 
from mmdet.models.builder import build_backbone, build_head, build_neck
import MinkowskiEngine as ME
from mmdet3d.core import bbox3d2result
from mmcv.runner import auto_fp16
from projects.mvsdetection.datasets.tsdf import TSDF, coordinates
from projects.mvsdetection.datasets.pipelines.fcaf3d_transforms import TransformFeaturesBBoxes, sample_mask

def backproject(voxel_dim, voxel_size, origin, projection, features):
    """ Takes 2d features and fills them along rays in a 3d volume

    This function implements eqs. 1,2 in https://arxiv.org/pdf/2003.10432.pdf
    Each pixel in a feature image corresponds to a ray in 3d.
    We fill all the voxels along the ray with that pixel's features.

    Args:
        voxel_dim: size of voxel volume to construct (nx,ny,nz)
        voxel_size: metric size of each voxel (ex: .04m)
        origin: origin of the voxel volume (xyz position of voxel (0,0,0))
        projection: bx4x3 projection matrices (intrinsics@extrinsics)
        features: bxcxhxw  2d feature tensor to be backprojected into 3d

    Returns:
        volume: b x c x nx x ny x nz 3d feature volume
        valid:  b x 1 x nx x ny x nz volume.
                Each voxel contains a 1 if it projects to a pixel
                and 0 otherwise (not in view frustrum of the camera)
    """

    batch = features.size(0)
    channels = features.size(1)
    device = features.device
    nx, ny, nz = voxel_dim

    coords = coordinates(voxel_dim, device).unsqueeze(0).expand(batch,-1,-1) # bx3xhwd
    world = coords.type_as(projection) * voxel_size + origin.to(device).unsqueeze(2)
    world = torch.cat((world, torch.ones_like(world[:,:1]) ), dim=1)
    
    camera = torch.bmm(projection, world)
    px = (camera[:,0,:]/camera[:,2,:]).round().type(torch.long)
    py = (camera[:,1,:]/camera[:,2,:]).round().type(torch.long)
    pz = camera[:,2,:]

    # voxels in view frustrum
    height, width = features.size()[2:]
    valid = (px >= 0) & (py >= 0) & (px < width) & (py < height) & (pz>0) # bxhwd

    # put features in volume
    volume = torch.zeros(batch, channels, nx*ny*nz, dtype=features.dtype, 
                         device=device)
    for b in range(batch):
        volume[b,:,valid[b]] = features[b,:,py[b,valid[b]], px[b,valid[b]]]

    volume = volume.view(batch, channels, nx, ny, nz)
    valid = valid.view(batch, 1, nx, ny, nz)

    return volume, valid

def get_ray_parameter(projection, features):
    """Get the ray(O + tD)'s parameters

    Args:
        projection: bx3x4 projection matrices (intrinsics@extrinsics)
        features: bxcxhxw  2d feature tensor to be backprojected into 3d
        
    Returns:
        o: b x 3 x (H x W), the origin of all the rays
        d: b x 3 x (H x W), the direction of all the rays, normalized 
    """
    B, C, H, W = features.size()
    device = features.device

    #2d grid of each pixel
    v, u = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    u = u.view(1, 1, H, W).repeat(B, 1, 1, 1).float().to(device) #B * 1 * H * W
    v = v.view(1, 1, H, W).repeat(B, 1, 1, 1).float().to(device) #B * 1 * H * W
    zero_depth = torch.zeros_like(u).float().to(device) #B * 1 * H * W
    one_depth = torch.ones_like(u).float().to(device) #B * 1 * H * W
    ones = torch.ones_like(u).float().to(device) #B * 1 * H * W
    uv_zeros = torch.cat((u * zero_depth, v * zero_depth, zero_depth, ones), dim=1).view(B, 4, H * W) #B * 4 * (H * W)
    uv_ones = torch.cat((u * one_depth, v * one_depth, one_depth, ones), dim=1).view(B, 4, H * W) #B * 4 * (H * W)
          
    #change projection to 4*4, and get its reverse
    ones_projection = torch.tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(B, 1, 1).float().to(device) #B * 1 * 4
    new_projection = torch.cat((projection, ones_projection), dim=1) #B * 4 * 4
    projection_inverse = []
    for i in range(B):
        inverse = torch.inverse(new_projection[i]) #4 * 4
        projection_inverse.append(inverse)
    projection_inverse = torch.stack(projection_inverse, dim=0) #B * 4 * 4
    
    #get the o and d of the ray 
    o = torch.bmm(projection_inverse, uv_zeros)
    o = o[:, 0:3, :]
    d = torch.bmm(projection_inverse, uv_ones)
    d = d[:, 0:3, :]
    d = d - o
    d = F.normalize(d, p=2, dim=1)
    return o, d




@DETECTORS.register_module()
class AtlasRayMarching(nn.Module):
    def __init__(self, 
                 pixel_mean, 
                 pixel_std, 
                 voxel_size, 
                 n_scales, 
                 voxel_dim_train, 
                 voxel_dim_test, 
                 origin, 
                 backbone2d_stride, 
                 backbone2d,
                 feature_2d, 
                 backbone_3d, 
                 tsdf_head, 
                 detection_backbone, 
                 detection_head, 
                 feature_transform,
                 save_path,
                 loss_weight_recon=1.0,
                 loss_weight_detection=1.0,
                 voxel_size_fcaf3d=0.01,
                 use_batchnorm_train=True,
                 use_batchnorm_test=True,
                 use_tsdf=False,
                 max_points=None,
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None):
        super(AtlasRayMarching, self).__init__()
        # networks
        self.fp16_enabled = False
        self.fpn = build_backbone(backbone2d)
        self.feature_2d = build_backbone(feature_2d)
        self.backbone3d = build_backbone(backbone_3d)
        self.tsdf_head = build_head(tsdf_head)
        self.detection_backbone = build_backbone(detection_backbone)
        self.detection_head = build_head(detection_head)
        
        if feature_transform != None:
            self.feature_transform = TransformFeaturesBBoxes(**feature_transform)
        else:
            self.feature_transform = None

        # other hparams
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.voxel_size = voxel_size
        self.n_scales = n_scales 
        self.voxel_dim_train = voxel_dim_train
        self.voxel_dim_test = voxel_dim_test
        self.voxel_size_fcaf3d = voxel_size_fcaf3d
        self.use_batchnorm_train = use_batchnorm_train
        self.use_batchnorm_test = use_batchnorm_test
        self.use_tsdf = use_tsdf
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path) 

        self.loss_weight_recon = loss_weight_recon
        self.loss_weight_detection = loss_weight_detection
        
        self.max_points = max_points

        self.origin = torch.tensor(origin).view(1,3)
        self.backbone2d_stride = backbone2d_stride
        self.initialize_volume()
                

    def initialize_volume(self):
        """ Reset the accumulators.
        
        self.volume is a voxel volume containg the accumulated features
        self.valid is a voxel volume containg the number of times a voxel has
            been seen by a camera view frustrum
        """
        self.volume = 0
        self.valid = 0
        self.volume_detection = 0
        self.valid_detection = 0

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def backbone2d(self, image):
        x = self.fpn(image)
        x = self.feature_2d(x)
        return x

    def aggregate_2d_features(self, projection, image=None, feature=None):
        """ Backprojects image features into 3D and accumulates them.

        This is the first half of the network which is run on every frame.
        Only pass one of image or feature. If image is passed 2D features
        are extracted from the image using self.backbone2d. When features
        are extracted external to this function pass features (used when 
        passing multiple frames through the backbone2d simultaniously
        to share BatchNorm stats).

        Args:
            projection: bx3x4 projection matrix
            image: bx3xhxw RGB image
            feature: bxcxh'xw' feature map (h'=h/stride, w'=w/stride)

        Feature volume is accumulated into self.volume and self.valid
        """

        assert ((image is not None and feature is None) or 
                (image is None and feature is not None))

        if feature is None:
            image = self.normalizer(image)
            feature = self.backbone2d(image)

        # backbone2d reduces the size of the images so we 
        # change intrinsics to reflect this
        projection = projection.clone()
        projection[:,:2,:] = projection[:,:2,:] / self.backbone2d_stride

        volume, valid = backproject(self.voxel_dim, self.voxel_size, self.origin, projection, feature)

        self.volume = self.volume + volume
        self.valid = self.valid + valid

    def clear_3d_features(self):
        """
        Clear the aggregated 3D features, remove nans
        """
        self.volume = self.volume / self.valid

        # remove nans (where self.valid==0)
        self.volume = self.volume.transpose(0,1)
        self.volume[:, self.valid.squeeze(1)==0] = 0
        self.volume = self.volume.transpose(0,1)
        self.valid = self.valid > 0
        
        
    def atlas_reconstruction(self, targets=None):
        """ Refines accumulated features and regresses output TSDF.

        This is the second half of the network. It should be run once after
        all frames have been accumulated. It may also be run more fequently
        to visualize incremental progress.

        Args:
            targets: used to compare network output to ground truth

        Returns:
            tuple of dicts ({outputs}, {losses})
                if targets is None, losses is empty
        """
        x = self.backbone3d(self.volume)
        output, loss = self.tsdf_head(x, targets)
        return output, loss 
    
    
    def fcaf3d_detection(self, inputs, tsdf, test=False):   
        sparse_coords, sparse_features, inputs['gt_bboxes_3d'] = self.switch_to_sparse(self.volume, self.valid, inputs['gt_bboxes_3d'], inputs['offset'], tsdf, test)
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(sparse_coords[i] / self.voxel_size_fcaf3d, sparse_features[i]) for i in range(len(sparse_features))], device=sparse_features[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.detection_backbone(x)        
        centernesses, bbox_preds, cls_scores, points = list(self.detection_head(x))        
        losses = self.detection_head.loss(centernesses, bbox_preds, cls_scores, points, inputs['gt_bboxes_3d'], inputs['gt_labels_3d'] )
        if test:
            self.detection_head.get_bboxes(centernesses, bbox_preds, cls_scores, points, inputs['scene'], self.save_path)
        return losses
        
    def switch_to_sparse(self, feature, valid, gt_bboxes, offsets, tsdf, test):
        '''
        Switch the volume to sparse mode, including data augmentation
        
        Args:
            inputs [dict]: [the dict of all the inputs]
            tsdf [torch float32 array], [B * 1 * X * Y * Z]: [the predicted tsdf volume]
            test [bool]: [whether in test mode]
        
        Returns:
            sparse_features [list of torch float32 array], [N * C]: [the list of sparse feature, N is the number of valid points]
        '''
        #get coords
        B, C, X, Y, Z = feature.shape 
        device = feature.device
        x_coord, y_coord, z_coord = torch.meshgrid(
            torch.arange(X, device=device), 
            torch.arange(Y, device=device), 
            torch.arange(Z, device=device))  #X * Y * Z
        x_coord = x_coord.view(1, 1, X, Y, Z).repeat(B, 1, 1, 1, 1)
        y_coord = y_coord.view(1, 1, X, Y, Z).repeat(B, 1, 1, 1, 1)
        z_coord = z_coord.view(1, 1, X, Y, Z).repeat(B, 1, 1, 1, 1)
        coords = torch.concat((x_coord, y_coord, z_coord), dim=1).float() #B * 3 * X * Y * Z
        coords = coords.permute(0, 2, 3, 4, 1).view(B, X * Y * Z, 3)
        for i in range(B):
            coords[i] = coords[i] * self.voxel_size + offsets[i]
        feature = feature.permute(0, 2, 3, 4, 1).view(B, X * Y * Z, C)
        #coords: B * (X * Y * Z) * 3
        #feature: B * (X * Y * Z) * C
        

        #mask and selection
        tsdf_sum = int(((tsdf < 0.999) & (tsdf > -0.999)).sum().detach().cpu().numpy())
        if self.use_tsdf:
            if tsdf_sum > 1000:
                mask = (tsdf < 0.999) & (tsdf > -0.999) & valid
            else:
                mask = valid 
                print('tsdf:', tsdf_sum)
        else:
            mask = valid
        
        if self.max_points != None:
            masks = []
            for i in range(B):
                masks.append(sample_mask(mask[i], self.max_points))
            mask = torch.stack(masks, dim=0)
        mask = mask.view(B, X * Y * Z)

        #selected coords
        selected_coords = []
        new_gt_bboxes = []
        for b in range(B):
            selected_coords_batch = []
            for i in range(3):
                selected_coord = torch.masked_select(coords[b, :, i], mask[b])
                selected_coords_batch.append(selected_coord)
            selected_coords_batch = torch.stack(selected_coords_batch, dim=1).float()
            if self.feature_transform != None and (not test):
                selected_coords_batch, gt_bbox = self.feature_transform(selected_coords_batch, gt_bboxes[b])
            else:
                gt_bbox = gt_bboxes[b]
            selected_coords.append(selected_coords_batch)
            new_gt_bboxes.append(gt_bbox)
        
        
        selected_features = []
        for b in range(B):
            selected_features_batch = []
            for i in range(C):
                selected_feature = torch.masked_select(feature[b, :, i], mask[b]) 
                selected_features_batch.append(selected_feature)
            selected_features_batch = torch.stack(selected_features_batch, dim=1).float()
            selected_features.append(selected_features_batch)
          
        return selected_coords, selected_features, new_gt_bboxes
        
    def forward_train(self, inputs):
        self.voxel_dim = self.voxel_dim_train
        self.initialize_volume()

        image = inputs['imgs']
        projection = inputs['projection']
        images = image.transpose(0,1)
        projections = projection.transpose(0,1)
        if self.use_batchnorm_train:
            image = images.reshape(images.shape[0]*images.shape[1], *images.shape[2:])
            image = self.normalizer(image)
            features = self.backbone2d(image)
            features = features.view(images.shape[0], images.shape[1], *features.shape[1:])
            for projection, feature in zip(projections, features):
                self.aggregate_2d_features(projection, feature=feature)
            self.clear_3d_features()
        else:
            for projection, image in zip(projections, images):
                self.aggregate_2d_features(projection, image=image)
            self.clear_3d_features()
        
        
        
        
        # run 3d cnn
        recon_result, recon_loss = self.atlas_reconstruction(inputs['tsdf_list'])
        
        #Ray marching
        i = 0
        for projection, feature in zip(projections, features):
            o, d, voxel_id, valid, tsdf_results, weights = self.ray_projection(projection, feature, recon_result['scene_tsdf_004'])
            
            kebab=0
            scene_id = inputs['scene'][0]
            image_id = inputs['image_ids'][0][i]
            result_tsdf = TSDF(self.voxel_size, self.origin, recon_result['scene_tsdf_004'][0].squeeze(0))
            result_mesh = result_tsdf.get_mesh()
            if not os.path.exists(os.path.join(self.save_path, scene_id + '_' + str(image_id))):
                os.makedirs(os.path.join(self.save_path, scene_id + '_' + str(image_id)))
            mesh_path = os.path.join(self.save_path, scene_id + '_' + str(image_id), scene_id + '_' + str(image_id) + '.ply')
            result_mesh.export(mesh_path)
            result_path = os.path.join(self.save_path, scene_id + '_' + str(image_id), scene_id + '_' + str(image_id) + '.npz')
            np.savez(result_path, o=o.detach().cpu().numpy(), d=d.detach().cpu().numpy(), voxel_id=voxel_id.detach().cpu().numpy(), valid=valid.detach().cpu().numpy(), tsdf_results=tsdf_results.detach().cpu().numpy(), origin=self.origin, weights=weights.detach().cpu().numpy())
            
            tsdf_path = os.path.join(self.save_path, scene_id + '_' + str(image_id), scene_id + '_' + str(image_id) + '_tsdf.npz')
            result_tsdf.save(tsdf_path)
            gt_tsdf = TSDF(self.voxel_size, self.origin, inputs['tsdf_list']['tsdf_gt_004'][0].squeeze(0))
            gt_tsdf_path = os.path.join(self.save_path, scene_id + '_' + str(image_id), scene_id + '_' + str(image_id) + '_gt_tsdf.npz')
            gt_tsdf.save(gt_tsdf_path)
            gt_mesh_path = os.path.join(self.save_path, scene_id + '_' + str(image_id), scene_id + '_' + str(image_id) + '_gt.ply')
            gt_mesh = gt_tsdf.get_mesh()
            gt_mesh.export(gt_mesh_path)

            i += 1

        
        detection_loss = self.fcaf3d_detection(inputs, recon_result['scene_tsdf_004'], test=False)        
                
        #get loss 
        losses = {}
        for key in recon_loss.keys():
            losses[key] = recon_loss[key] * self.loss_weight_recon
        for key in detection_loss.keys():
            losses[key] = detection_loss[key] * self.loss_weight_detection

        return losses
    


    
    def forward_test(self, inputs):       
        self.voxel_dim = self.voxel_dim_test
        self.initialize_volume()

        image = inputs['imgs']
        projection = inputs['projection']
        images = image.transpose(0,1)
        projections = projection.transpose(0,1)
        if self.use_batchnorm_test:
            image = images.reshape(images.shape[0]*images.shape[1], *images.shape[2:])
            image = self.normalizer(image)
            features = self.backbone2d(image)
            features = features.view(images.shape[0], images.shape[1], *features.shape[1:])
            for projection, feature in zip(projections, features):
                self.aggregate_2d_features(projection, feature=feature)
            self.clear_3d_features()
        else:
            for projection, image in zip(projections, images):
                self.aggregate_2d_features(projection, image=image)
            self.clear_3d_features()
        
        # run 3d cnn
        recon_result, recon_loss = self.atlas_reconstruction(inputs['tsdf_list'])
        

        
        detection_loss = self.fcaf3d_detection(inputs, recon_result['scene_tsdf_004'], test=True)        

        #get loss 
        losses = {}
        for key in recon_loss.keys():
            losses[key] = recon_loss[key] * self.loss_weight_recon
        for key in detection_loss.keys():
            losses[key] = detection_loss[key] * self.loss_weight_detection
        
        print(losses)
        recon_results = self.post_process(recon_result, inputs)
        for result in recon_results:
            scene_id = result['scene']
            tsdf_pred = result['scene_tsdf']
            mesh_pred = tsdf_pred.get_mesh()
            if not os.path.exists(os.path.join(self.save_path, scene_id)):
                os.makedirs(os.path.join(self.save_path, scene_id))
            tsdf_pred.save(os.path.join(self.save_path, scene_id, scene_id + '.npz'))
            mesh_pred.export(os.path.join(self.save_path, scene_id, scene_id + '.ply'))
            #kebab = result['kebab'].get_mesh()
            #kebab.export(os.path.join(self.save_path, scene_id, scene_id + '_gt.ply'))
        
       # self.test_transform_valid(inputs)
        return [{}]

    def post_process(self, outputs, inputs):
        key = 'scene_tsdf_004'
        outs = []
        batch_size = len(outputs[key])
        #batch_size=1

        for i in range(batch_size):
            scene_id = inputs['scene'][i]
            tsdf = TSDF(self.voxel_size, self.origin, outputs[key][i].squeeze(0))
            offset = inputs['offset'][i].view(1, 3)
            tsdf.origin = offset
            
            out = {}
            out['scene'] = scene_id
            out['scene_tsdf'] = tsdf
            
            kebab = TSDF(self.voxel_size, self.origin, inputs['tsdf_list']['tsdf_gt_004'][i].squeeze(0))
            kebab.origin = offset
            out['kebab'] = kebab
            outs.append(out)

        return outs
    
    
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

    @auto_fp16()
    def forward(self, return_loss=True, rescale=False, **kwargs):
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
            return self.forward_train(inputs)
        else:
            inputs = self.data_converter(kwargs)
            return self.forward_test(inputs)
        
    def data_converter(self, data):
        data['imgs'] = torch.stack(data['imgs'], dim=0)
        data['projection'] = torch.stack(data['projection'], dim=0)
        if 'axis_align_matrix' in data.keys():
            data['axis_align_matrix'] = torch.stack(data['axis_align_matrix'], dim=0)

        device = data['gt_labels_3d'][0].device 
        for i in range(len(data['gt_bboxes_3d'])):
            data['gt_bboxes_3d'][i] = data['gt_bboxes_3d'][i].to(device)

        if 'offset' in data.keys():
            data['offset'] = torch.stack(data['offset'], dim=0)

        batch_size = len(data['tsdf_dict'])
        voxel_sizes = [int(key[8:]) for key in data['tsdf_dict'][0].keys()] #4, 8, 16
        real_tsdf_list = {}
        for voxel_size in voxel_sizes:
            tsdf_list = []
            tsdf_key = 'tsdf_gt_' + str(voxel_size).zfill(3) #004, 008, 016
            for j in range(batch_size):
                tsdf_list.append(data['tsdf_dict'][j][tsdf_key].tsdf_vol.unsqueeze(0)) #1 * X * Y * Z
            tsdf_list = torch.stack(tsdf_list, dim=0) #B * 1 * X * Y * Z
            tsdf_list = tsdf_list.to(device) 
            real_tsdf_list[tsdf_key] = tsdf_list 
        data['tsdf_list'] = real_tsdf_list
        data.pop('axis_align_matrix')
        data.pop('tsdf_dict')
        return data 
    
    def init_weights(self):
        pass

    def ray_projection(self, projection, features, tsdf, grids=300, weight_threshold=0.1):
        """ Get the 3D coordinates of each pixel with tsdf
        Args:
            voxel_dim: size of voxel volume to construct (nx,ny,nz)
            voxel_size: metric size of each voxel (ex: .04m)
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
            projection: bx3x4 projection matrices (intrinsics@extrinsics)
            features: bxcxhxw  2d feature tensor to be backprojected into 3d
            tsdf: b x 1 x nx x ny x nz tsdf map
            grids: the grid samples of each line
            weight_threshold: the min threshold for a voxel to be considered

        Returns:
            volume: b x c x nx x ny x nz 3d feature volume
            valid:  b x 1 x nx x ny x nz volume.
                    Each voxel contains a 1 if it projects to a pixel
                    and 0 otherwise (not in view frustrum of the camera)
        """
    
        X, Y, Z = self.voxel_dim
        B, C, H, W = features.size()
        N = grids
        device = features.device
        origin_extend = self.origin.view(B, 3, 1).repeat(1, 1, H * W).to(device) #B * 3 * (H * W)
        projection = projection.clone()
        projection[:,:2,:] = projection[:,:2,:] / self.backbone2d_stride
    
        o, d = get_ray_parameter(projection, features) #B * 3 * (H * W)
    
        #get the t
        t_max = sqrt(X ** 2 + Y ** 2 + Z ** 2) * self.voxel_size 
        t_one = t_max / N
        t = []
        for j in range(grids):
            current_t = N * j 
            current_ts = torch.ones(B, 3, H * W, 1, dtype=torch.float32) * current_t
            t.append(current_ts)
        t = torch.concat(t, dim=3).view(B, 3, H * W * N).to(device) #B * 3 * (H * W * N)
        
        #get voxel ids
        d = d.view(B, 3, H * W, 1).repeat(1, 1, 1, N).view(B, 3, H * W * N) #B * 3 * (H * W * N)
        o = o.view(B, 3, H * W, 1).repeat(1, 1, 1, N).view(B, 3, H * W * N) #B * 3 * (H * W * N)
        origin_extend = origin_extend.view(B, 3, H * W, 1).repeat(1, 1, 1, N).view(B, 3, H * W * N) #B * 3 * (H * W * N)
        place = o + d * t
        voxel_id = ((place - origin_extend) / self.voxel_size).round().type(torch.long) #B * 3 * (H * W * N)
        valid = (voxel_id[:, 0, :] >= 0) & (voxel_id[:, 0, :] < X) & \
                (voxel_id[:, 1, :] >= 0) & (voxel_id[:, 1, :] < Y) & \
                (voxel_id[:, 2, :] >= 0) & (voxel_id[:, 2, :] < Z) #B * (H * W * N)
        voxel_id[:, 0, :][valid == 0] = 0
        voxel_id[:, 1, :][valid == 0] = 0
        voxel_id[:, 2, :][valid == 0] = 0

    
        #get tsdf values
        tsdf_results = []
        for b in range(B):
            tsdf_result = tsdf[b, 0, voxel_id[b, 0, :], voxel_id[b, 1, :], voxel_id[b, 2, :]]
            tsdf_results.append(tsdf_result)
        tsdf_results = torch.stack(tsdf_results, dim=0) #B * (H * W * N)
        tsdf_results[valid == 0] = 1.0 #mask bad tsdf
        tsdf_results = tsdf_results.view(B, H, W, N) #B * H * W * N 
    
        o = o.view(B, 3, H, W, N)
        d = d.view(B, 3, H, W, N)
        voxel_id = voxel_id.view(B, 3, H, W, N)
        valid = valid.view(B, H, W, N)
        
        #get the weights of each grids based on NEUS
        sigmoid_tsdf = torch.sigmoid(-tsdf_results)
        sigmoid_tsdf_next = torch.concat((sigmoid_tsdf[:, :, :, 1:], sigmoid_tsdf[:, :, :, -1:]), dim=3)
        alpha = torch.clamp((sigmoid_tsdf - sigmoid_tsdf_next) / sigmoid_tsdf, min=0) #B * H * W * N
        
        
        #normalize to sum=1
        weights = weights / torch.sum(weights, dim=3).view(B, H, W, 1).repeat(1, 1, 1, N)
        
        #get the features of each grids, add weights
        weighted_features = features.view(B, C, H, W, 1).repeat(1, 1, 1, 1, N)
        valid_grid = weights >= weight_threshold
        valid_final = valid & valid_grid
        weights = weights * valid_final
        weights_repeat = weights.view(B, 1, H, W, N).repeat(1, C, 1, 1, 1)
        weighted_features = weighted_features * weights_repeat
        
        '''
        #allocate the weights to voxels
        volume = torch.zeros(B, C, X, Y, Z, dtype=features.dtype, device=device)
        voxel_id = voxel_id.view(B, 3, H * W * N)
        weighted_features = weighted_features.view(B, C, H * W * N)
        for b in range(b):
            for c in range(C):
                volume[b, c, voxel_id[b, 0, :], voxel_id[b, 1, :], voxel_id[b, 2, :]] = weighted_features[b, c, :]
        valid_output = volume > 0
        return volume, valid_output
        '''
        
        #可视化方案：把voxel id转化成实际位置，可视化显示出来
        return o, d, voxel_id, valid, tsdf_results, weights 
    