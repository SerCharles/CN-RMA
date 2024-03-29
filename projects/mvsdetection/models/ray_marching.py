'''Our main model for CN-RMA
'''

import numpy as np
import os
from math import *
import open3d as o3d
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
from projects.mvsdetection.datasets.pipelines.fcaf3d_transforms import TransformFeaturesBBoxes, sample_mask, sample_points

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
class RayMarching(nn.Module):
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
                 max_points=None,
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 use_feature_transform=True,
                 ray_marching_type='neus',
                 depth_points=None,
                 neus_threshold=None,
                 middle_save_path=None,
                 middle_visualize_path=None):
        super(RayMarching, self).__init__()
        # networks
        self.fp16_enabled = False
        self.fpn = build_backbone(backbone2d)
        self.feature_2d = build_backbone(feature_2d)
        self.backbone3d = build_backbone(backbone_3d)
        self.tsdf_head = build_head(tsdf_head)
        self.detection_backbone = build_backbone(detection_backbone)
        self.detection_head = build_head(detection_head)
        
        if use_feature_transform != True:
            feature_transform = None
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
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path) 

        self.loss_weight_recon = loss_weight_recon
        self.loss_weight_detection = loss_weight_detection
        
        self.max_points = max_points

        self.origin = torch.tensor(origin).view(1,3)
        self.backbone2d_stride = backbone2d_stride
        self.initialize_volume()
        
        self.ray_marching_type = ray_marching_type
        self.neus_threshold = neus_threshold
        self.depth_points = depth_points
        if self.ray_marching_type == 'neus':
            assert self.neus_threshold != None 
        elif self.ray_marching_type == 'depth':
            assert self.depth_points in [1, 2, 3, 4]

        self.middle_save_path = middle_save_path
        self.middle_visualize_path = middle_visualize_path


    def initialize_volume(self):
        """ Reset the accumulators.
        
        self.volume is a voxel volume containg the accumulated features
        self.valid is a voxel volume containg the number of times a voxel has
            been seen by a camera view frustrum
        """
        self.volume = 0
        self.valid = 0
        self.points_detection = []

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def backbone2d(self, image):
        x = self.fpn(image)
        x = self.feature_2d(x)
        return x

    def aggregate_2d_features(self, projection, feature):
        """ Backprojects image features into 3D and accumulates them.

        This is the first half of the network which is run on every frame.
        Only pass one of image or feature. If image is passed 2D features
        are extracted from the image using self.backbone2d. When features
        are extracted external to this function pass features (used when 
        passing multiple frames through the backbone2d simultaniously
        to share BatchNorm stats).

        Args:
            projection: bx3x4 projection matrix
            feature: bxcxh'xw' feature map (h'=h/stride, w'=w/stride)

        Feature volume is accumulated into self.volume and self.valid
        """
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
        
     
    def aggregate_2d_features_ray_marching(self, projections, features, tsdf):
        """ Backprojects image features into 3D using ray marching and get the final result

        Args:
            projection: bx3x4 projection matrix
            feature: bxcxh'xw' feature map (h'=h/stride, w'=w/stride)

        Feature volume is accumulated into self.volume and self.valid
        """

        B = projections.shape[1]
        for b in range(B):
            self.points_detection.append(None)
        
        for projection, feature in zip(projections, features):
            projection = projection.clone()
            projection[:,:2,:] = projection[:,:2,:] / self.backbone2d_stride
            try:
                if self.ray_marching_type == 'neus':
                    points = self.ray_projection_neus(projection, feature, tsdf, weight_threshold=self.neus_threshold)
                elif self.ray_marching_type == 'depth':
                    points = self.ray_projection_depth(projection, feature, tsdf, select_grids=self.depth_points)
            except:
                points = None
            
            if points == None:
                print('No valid points!')
                continue
            else:
                for b in range(len(points)):
                    if self.points_detection[b] == None:
                        self.points_detection[b] = points[b]
                    else:
                        self.points_detection[b] = torch.concat((self.points_detection[b], points[b]), dim = 0) #M * C



        #add weights        
        weighted_points = []
        for b in range(B):
            places = self.points_detection[b][:, 0:3] #M * 3
            weights = self.points_detection[b][:, 3:4] #M
            point_features = self.points_detection[b][:, 4:] #M * C
            weights = weights / torch.mean(weights) #M 
            weighted_features = point_features * weights #M * C
            current_points = torch.concat((places, weighted_features), dim=1) #M * (C + 3)
            weighted_points.append(current_points)
        self.points_detection = weighted_points
        

            

        
    def atlas_reconstruction(self, targets=None):
        """main function of atlas reconstruction given the volume
        """
        x = self.backbone3d(self.volume)
        output, loss = self.tsdf_head(x, targets)
        return output, loss 
    
    

    def fcaf3d_detection(self, inputs, points, test=False):   
        '''main function for fcaf3d detection given pointcloud with features
        ''' 
        sparse_coords, sparse_features, inputs['gt_bboxes_3d'] = self.switch_pointcloud(points, inputs['gt_bboxes_3d'], inputs['offset'], test)
        
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(sparse_coords[i] / self.voxel_size_fcaf3d, sparse_features[i]) for i in range(len(sparse_features))], device=sparse_features[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.detection_backbone(x)        
        centernesses, bbox_preds, cls_scores, points = list(self.detection_head(x))        
        losses = self.detection_head.loss(centernesses, bbox_preds, cls_scores, points, inputs['gt_bboxes_3d'], inputs['gt_labels_3d'] )
        if test:
            self.detection_head.get_bboxes(centernesses, bbox_preds, cls_scores, points, inputs['scene'], self.save_path)
        return losses
        

    def switch_pointcloud(self, points, gt_bboxes, offsets, test):
        '''
        Switch the volume to sparse mode, including data augmentation
        
        Args:
            points [array of torch float32 array], [N * (3 + C) each]: [the points of 3D features, N is the number of points]
            gt_bboxes [array of torch float array], [M * 7 each]: [the gt bboxes, M is the number of gt bboxes]
            offsets [array of torch float array], [3 each]: [the offsets of each scene]
            test [bool]: [whether in test mode]
        
        Returns:
            sparse_coords [list of torch float32 array], [N * 3]: [the list of sparse coords, N is the number of valid points]
            sparse_features [list of torch float32 array], [N * C]: [the list of sparse feature, N is the number of valid points]
            new_gt_bboxes [array of torch float array], [M * 7 each]: [the gt bboxes, M is the number of gt bboxes]
        '''
        B = len(points)
        device = points[0].device
        C = points[0].shape[1] - 3
        masks = []
        coords = []
        features = []
        for b in range(B):
            #get new places
            coord = points[b][:, 0:3]
            feature = points[b][:, 3:]
            coord = coord + offsets[b]
            coords.append(coord)
            features.append(feature)
            
            #get sample mask
            if self.max_points != None:
                mask = sample_points(coord, max_points=self.max_points)
                masks.append(mask)
        if self.max_points == None:
            masks = None

        #selected coords
        selected_coords = []
        new_gt_bboxes = []
        for b in range(B):
            if masks != None:
                selected_coords_batch = []
                for i in range(3):
                    selected_coord = torch.masked_select(coords[b][:, i], masks[b])
                    selected_coords_batch.append(selected_coord)
                selected_coords_batch = torch.stack(selected_coords_batch, dim=1).float()
            else:
                selected_coords_batch = coords[b]
            if self.feature_transform != None and (not test):
                selected_coords_batch, gt_bbox = self.feature_transform(selected_coords_batch, gt_bboxes[b])
            else:
                gt_bbox = gt_bboxes[b]
            selected_coords.append(selected_coords_batch)
            new_gt_bboxes.append(gt_bbox)
        
        
        selected_features = []
        for b in range(B):
            if masks != None:
                selected_features_batch = []
                for i in range(C):
                    selected_feature = torch.masked_select(features[b][:, i], masks[b]) 
                    selected_features_batch.append(selected_feature)
                selected_features_batch = torch.stack(selected_features_batch, dim=1).float()
            else:
                selected_features_batch = features[b]
            selected_features.append(selected_features_batch)
          
        return selected_coords, selected_features, new_gt_bboxes

    def forward_train(self, inputs):
        '''Main train function
        ''' 
        self.voxel_dim = self.voxel_dim_train
        self.initialize_volume()
        scene_id = inputs['scene'][0]
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
                self.aggregate_2d_features(projection, feature)
            self.clear_3d_features()
        else:
            features = []
            for image in images:
                image = self.normalizer(image)
                feature = self.backbone2d(image)
                features.append(feature)
            features = torch.stack(features, dim=0)
            for projection, feature in zip(projections, features):
                self.aggregate_2d_features(projection, feature)
            self.clear_3d_features()
        
        # run 3d cnn
        recon_result, recon_loss = self.atlas_reconstruction(inputs['tsdf_list'])
        self.aggregate_2d_features_ray_marching(projections, features, recon_result['scene_tsdf_004'])
        detection_loss = self.fcaf3d_detection(inputs, self.points_detection, test=False)

        
        #get loss 
        losses = {}
        for key in recon_loss.keys():
            losses[key] = recon_loss[key] * self.loss_weight_recon
        for key in detection_loss.keys():
            losses[key] = detection_loss[key] * self.loss_weight_detection

        return losses
    


    
    def forward_test(self, inputs):      
        '''Main test function
        ''' 
        self.voxel_dim = self.voxel_dim_test
        self.initialize_volume()
        scene_id = inputs['scene'][0]
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
            features = []
            for image in images:
                image = self.normalizer(image)
                feature = self.backbone2d(image)
                features.append(feature)
            features = torch.stack(features, dim=0)
            for projection, feature in zip(projections, features):
                self.aggregate_2d_features(projection, feature)
            self.clear_3d_features()
        
        # run 3d cnn
        recon_result, recon_loss = self.atlas_reconstruction(inputs['tsdf_list'])
        
        
        #ray marching
        self.aggregate_2d_features_ray_marching(projections, features, recon_result['scene_tsdf_004'])
        detection_loss = self.fcaf3d_detection(inputs, self.points_detection, test=True)        

        #get loss 
        losses = {}
        for key in recon_loss.keys():
            losses[key] = recon_loss[key] * self.loss_weight_recon
        for key in detection_loss.keys():
            losses[key] = detection_loss[key] * self.loss_weight_detection
        
        print(losses)
        

        try:
            recon_results = self.post_process(recon_result, inputs)
            for result in recon_results:
                scene_id = result['scene']
                tsdf_pred = result['scene_tsdf']
                mesh_pred = tsdf_pred.get_mesh()
                if not os.path.exists(os.path.join(self.save_path, scene_id)):
                    os.makedirs(os.path.join(self.save_path, scene_id))
                tsdf_pred.save(os.path.join(self.save_path, scene_id, scene_id + '.npz'))
                mesh_pred.export(os.path.join(self.save_path, scene_id, scene_id + '.ply'))
        
            if self.middle_save_path != None:
                self.save_middle_result(scene_id, self.points_detection[0], result['scene_tsdf'].origin, self.middle_save_path, self.middle_visualize_path)
        
        except:
            scene_id = inputs['scene'][0]
            print(scene_id + 'is invalid!')
        
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
        '''Convert the data input
        '''
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

    def ray_projection_neus(self, projection, features, tsdf, grids=300, weight_threshold=None):
        """ Get the 3D coordinates of each pixel with NEUS
        Args:
            projection: bx3x4 projection matrices (intrinsics@extrinsics)
            features: bxcxhxw  2d feature tensor to be backprojected into 3d
            tsdf: b x 1 x nx x ny x nz tsdf map
            grids: the grid samples of each line
            weight_threshold: the min threshold for a voxel to be considered

        Returns:
            results: array of M * (C + 4) point clouds; concated by [places (B * M * 3), weights (B * M * 1), features(B * M * C)]
            if no possible point cloud, return None
        """   
        X, Y, Z = self.voxel_dim
        B, C, H, W = features.size()
        N = grids
        device = features.device
        
        with torch.no_grad():
            #get o, d, t
            origin_extend = self.origin.view(B, 3, 1).repeat(1, 1, H * W).to(device) #B * 3 * (H * W)
            o, d = get_ray_parameter(projection, features) #B * 3 * (H * W)
            
            t_max = sqrt(X ** 2 + Y ** 2 + Z ** 2) * self.voxel_size 
            t_one = t_max / N
            #t_one = 0.04
            current_ts = torch.arange(0, N, step=1, dtype=torch.float32).to(device)
            current_ts = current_ts * t_one 
            t = current_ts.view(1, 1, 1, N).repeat(B, 3, H * W, 1).view(B, 3, H * W * N)

            
            #get pixel ids
            v, u = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
            u = u.view(1, 1, H, W, 1).type(torch.long).to(device) 
            v = v.view(1, 1, H, W, 1).type(torch.long).to(device)
            pixel_ids = torch.concat((u, v), dim=1) 
            pixel_ids = pixel_ids.repeat(B, 1, 1, 1, N).view(B, 2, H * W * N)
                
            #get voxel ids
            d = d.view(B, 3, H * W, 1).repeat(1, 1, 1, N).view(B, 3, H * W * N) #B * 3 * (H * W * N)
            o = o.view(B, 3, H * W, 1).repeat(1, 1, 1, N).view(B, 3, H * W * N) #B * 3 * (H * W * N)
            origin_extend = origin_extend.view(B, 3, H * W, 1).repeat(1, 1, 1, N).view(B, 3, H * W * N) #B * 3 * (H * W * N)
            places = o + d * t #B * 3 * (H * W * N)
            voxel_ids = ((places - origin_extend) / self.voxel_size).round().type(torch.long) #B * 3 * (H * W * N)
            valid = (voxel_ids[:, 0, :] >= 0) & (voxel_ids[:, 0, :] < X) & \
                    (voxel_ids[:, 1, :] >= 0) & (voxel_ids[:, 1, :] < Y) & \
                    (voxel_ids[:, 2, :] >= 0) & (voxel_ids[:, 2, :] < Z) #B * (H * W * N)
            voxel_ids[:, 0, :][valid == 0] = 0
            voxel_ids[:, 1, :][valid == 0] = 0
            voxel_ids[:, 2, :][valid == 0] = 0
    
            #get tsdf values
            tsdf_results = []
            for b in range(B):
                tsdf_result = tsdf[b, 0, voxel_ids[b, 0, :], voxel_ids[b, 1, :], voxel_ids[b, 2, :]]
                tsdf_results.append(tsdf_result)
            tsdf_results = torch.stack(tsdf_results, dim=0) #B * (H * W * N)
            tsdf_results[valid == 0] = 1.0 #mask bad tsdf
            tsdf_results = tsdf_results.view(B, H, W, N) #B * H * W * N 
    
        o = o.view(B, 3, H, W, N)
        d = d.view(B, 3, H, W, N)
        places = places.view(B, 3, H, W, N)
        voxel_ids = voxel_ids.view(B, 3, H, W, N)
        valid = valid.view(B, H, W, N)
        pixel_ids = pixel_ids.view(B, 2, H, W, N)
            
        #get the weights of each grids based on NEUS
        #alpha_i = max((sigmoid(TSDF_i) - sigmoid(TSDF_i+1)) / sigmoid(TSDF_i), 0)
        #T_i = (1 - alpha_0) * (1 - alpha_1) * ...... * (1 - alpha_i-1)
        sigmoid_tsdf = torch.sigmoid(-tsdf_results)
        sigmoid_tsdf_next = torch.cat((sigmoid_tsdf[:, :, :, 1:], sigmoid_tsdf[:, :, :, -1:]), dim=3)
        alpha = torch.clamp((sigmoid_tsdf - sigmoid_tsdf_next) / sigmoid_tsdf, min=0) #B * H * W * N
        T_next = torch.cumprod(1 - alpha, dim=3)
        one = torch.ones((B, H, W, 1), dtype=torch.float32).to(device)
        T = torch.cat((one, T_next[:, :, :, :-1]), dim=3)
        weights = T * alpha
            
        valid_grid = weights >= weight_threshold
        valid_final = valid & valid_grid
        weights = weights * valid_final
                
        #select useful weights, places and pixel_ids
        flatten_valid = valid_final.view(B, H * W * N)
        voxel_ids = voxel_ids.view(B, 3, H * W * N)
        pixel_ids = pixel_ids.view(B, 2, H * W * N)
        places = places.view(B, 3, H * W * N)
        weights = weights.view(B, 1, H * W * N)
        useful_ids = []
        useful_weights = []
        useful_places = []
        useful_pixel_ids = []
            
        for b in range(B):
            useful_id = torch.squeeze(torch.nonzero(flatten_valid[b]))
            if len(useful_id) == 0:
                return None
                
            useful_weight = weights[b, :, useful_id].permute(1, 0) #M * 1
            useful_place = places[b, :, useful_id].permute(1, 0) #M * 3
            useful_pixel_id = pixel_ids[b, :, useful_id].permute(1, 0) #M * 2
            useful_ids.append(useful_id)
            useful_weights.append(useful_weight)
            useful_places.append(useful_place)
            useful_pixel_ids.append(useful_pixel_id)
                        
        #get useful features
        useful_features = []
        for b in range(B):
            useful_id = useful_ids[b]
            u = useful_pixel_ids[b][:, 0] #M
            v = useful_pixel_ids[b][:, 1] #M 
            useful_feature = features[b, :, v, u].permute(1, 0) #M * C
            useful_features.append(useful_feature)
                
        #concat all things
        results = []
        for b in range(B):
            result = torch.concat((useful_places[b], useful_weights[b], useful_features[b]), dim=1) #M * (C + 4)
            results.append(result)
        return results

    def ray_projection_depth(self, projection, features, tsdf, grids=300, select_grids=None):
        """ Get the 3D coordinates of each pixel with depth prediction
        Args:
            projection: bx3x4 projection matrices (intrinsics@extrinsics)
            features: bxcxhxw  2d feature tensor to be backprojected into 3d
            tsdf: b x 1 x nx x ny x nz tsdf map
            grids: the grid samples of each line
            select_grids: select 2*k points based on depth prediction, the weights decreases linearly

        Returns:
            results: array of M * (C + 4) point clouds; concated by [places (B * M * 3), weights (B * M * 1), features(B * M * C)]
            if no possible point cloud, return None
        """   
        X, Y, Z = self.voxel_dim
        B, C, H, W = features.size()
        N = grids
        device = features.device
                
        #get o, d, t
        origin_extend = self.origin.view(B, 3, 1).repeat(1, 1, H * W).to(device) #B * 3 * (H * W)
        o, d = get_ray_parameter(projection, features) #B * 3 * (H * W)
            
        t_max = sqrt(X ** 2 + Y ** 2 + Z ** 2) * self.voxel_size 
        t_one = t_max / N
        current_ts = torch.arange(0, N, step=1, dtype=torch.float32).to(device)
        current_ts = current_ts * t_one 
        t = current_ts.view(1, 1, 1, N).repeat(B, 3, H * W, 1).view(B, 3, H * W * N)


                
        #get voxel ids
        d = d.view(B, 3, H * W, 1).repeat(1, 1, 1, N).view(B, 3, H * W * N) #B * 3 * (H * W * N)
        o = o.view(B, 3, H * W, 1).repeat(1, 1, 1, N).view(B, 3, H * W * N) #B * 3 * (H * W * N)
        origin_extend = origin_extend.view(B, 3, H * W, 1).repeat(1, 1, 1, N).view(B, 3, H * W * N) #B * 3 * (H * W * N)
        places = o + d * t #B * 3 * (H * W * N)
        voxel_ids = ((places - origin_extend) / self.voxel_size).round().type(torch.long) #B * 3 * (H * W * N)
        valid = (voxel_ids[:, 0, :] >= 0) & (voxel_ids[:, 0, :] < X) & \
                (voxel_ids[:, 1, :] >= 0) & (voxel_ids[:, 1, :] < Y) & \
                (voxel_ids[:, 2, :] >= 0) & (voxel_ids[:, 2, :] < Z) #B * (H * W * N)
        voxel_ids[:, 0, :][valid == 0] = 0
        voxel_ids[:, 1, :][valid == 0] = 0
        voxel_ids[:, 2, :][valid == 0] = 0
    
        #get tsdf values
        tsdf_results = []
        for b in range(B):
            tsdf_result = tsdf[b, 0, voxel_ids[b, 0, :], voxel_ids[b, 1, :], voxel_ids[b, 2, :]]
            tsdf_results.append(tsdf_result)
        tsdf_results = torch.stack(tsdf_results, dim=0) #B * (H * W * N)
        tsdf_results[valid == 0] = 1.0 #mask bad tsdf
        tsdf_results = tsdf_results.view(B, H, W, N) #B * H * W * N 
    
        #get pixel ids
        v, u = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        u = u.view(1, 1, H, W).type(torch.long).to(device) 
        v = v.view(1, 1, H, W).type(torch.long).to(device)
        pixel_ids = torch.concat((u, v), dim=1).repeat(B, 1, 1, 1) #B * 2 * H * W
            
    
        o = o.view(B, 3, H, W, N)
        d = d.view(B, 3, H, W, N)
        places = places.view(B, 3, H, W, N)
        voxel_ids = voxel_ids.view(B, 3, H, W, N)
        valid = valid.view(B, H, W, N)
            
                        
        #get depth and weights
        tsdf_multiply = tsdf_results[:, :, :, :-1] * tsdf_results[:, :, :, 1:]
        tsdf_ones = torch.ones((B, H, W, 1), dtype=torch.float32).to(device)
        tsdf_multiply = torch.concat((tsdf_multiply, tsdf_ones), dim=3) #B * H * W * N
        tsdf_change = (tsdf_multiply <= 0).float() #TSDF change from >=0 to <=0, or from <=0 to >=0 
        best_index = torch.argmax(tsdf_change, dim=3) #B * H * W
        best_mask = torch.sum(tsdf_change, dim=3) > 0  #B * H * W
        best_weight = best_mask.float()
            
        #get other adjacent places
        if select_grids > 0:
            NUM = 2 * select_grids
            selected_indices = best_index.view(B, H, W, 1).repeat(1, 1, 1, NUM)
            selected_weights = best_weight.view(B, H, W, 1).repeat(1, 1, 1, NUM)
            add_num = torch.arange(0, NUM) - select_grids + 1
            add_num = add_num.to(device)
            multi_weight = torch.arange(0, select_grids) + 1
            multi_weight_ = multi_weight.clone().flip(dims=[0])
            multi_weight = torch.concat((multi_weight, multi_weight_), dim=0)
            multi_weight = (multi_weight.float() / select_grids).to(device)
            add_num = add_num.view(1, 1, 1, NUM).repeat(B, H, W, 1)
            multi_weight = multi_weight.view(1, 1, 1, NUM).repeat(B, H, W, 1)
            selected_indices = selected_indices + add_num
            selected_weights = selected_weights * multi_weight
            selected_mask = (selected_indices >= 0) & (selected_indices < N)
            selected_weights = selected_weights * selected_mask
            
            selected_valids = selected_weights > 0  #B * H * W * NUM
            selected_pixel_ids = pixel_ids.view(B, 2, H, W, 1).repeat(1, 1, 1, 1, NUM) #B * 2 * H * W * NUM
            selected_o = o[:, :, :, :, 0:NUM] #B * 3 * H * W * NUM
            selected_d = d[:, :, :, :, 0:NUM] #B * 3 * H * W * NUM
            selected_places = selected_o + selected_d * selected_indices * t_one #B * 3 * H * W * NUM
        else:
            NUM = 1
            selected_pixel_ids = pixel_ids.view(B, 2, H, W, 1)
            selected_o = o[:, :, :, :, 0:1] #B * 3 * H * W * 1
            selected_d = d[:, :, :, :, 0:1] #B * 3 * H * W * 1
            selected_indices = best_index.view(B, H, W, 1) + 0.5
            selected_weights = best_weight.view(B, H, W, 1)
            selected_valids = selected_weights > 0
            selected_places = selected_o + selected_d * selected_indices * t_one #B * 3 * H * W * 1    
            
            
                
        #select useful weights, places and pixel_ids
        flatten_valid = selected_valids.view(B, H * W * NUM)
        pixel_ids = selected_pixel_ids.view(B, 2, H * W * NUM)
        places = selected_places.view(B, 3, H * W * NUM)
        weights = selected_weights.view(B, 1, H * W * NUM)
        useful_ids = []
        useful_weights = []
        useful_places = []
        useful_pixel_ids = []
            
        for b in range(B):
            useful_id = torch.squeeze(torch.nonzero(flatten_valid[b]))
            if len(useful_id) == 0:
                return None
            useful_weight = weights[b, :, useful_id].permute(1, 0) #M * 1
            useful_place = places[b, :, useful_id].permute(1, 0) #M * 3
            useful_pixel_id = pixel_ids[b, :, useful_id].permute(1, 0) #M * 2
            useful_ids.append(useful_id)
            useful_weights.append(useful_weight)
            useful_places.append(useful_place)
            useful_pixel_ids.append(useful_pixel_id)
                  
        #get useful features
        useful_features = []
        for b in range(B):
            useful_id = useful_ids[b]
            u = useful_pixel_ids[b][:, 0] #M
            v = useful_pixel_ids[b][:, 1] #M 
            useful_feature = features[b, :, v, u].permute(1, 0) #M * C
            useful_features.append(useful_feature)
                
        #concat all things
        results = []
        for b in range(B):
            result = torch.concat((useful_places[b], useful_weights[b], useful_features[b]), dim=1) #M * (C + 4)
            results.append(result)
        
        return results


    def save_middle_result(self, scene_id, coords, offset, save_path, visualize_path):
        '''
        Save ray marching middle results for pretraining FCAF3D model
        '''
        N = coords.shape[0]
        C = coords.shape[1] - 3        
        coords = coords.detach().cpu()
        coords[:, 0:3] = coords[:, 0:3] + offset.detach().cpu()
        
        if N > self.max_points:
            choices = np.random.choice(N, self.max_points, replace=False)
            mask = np.zeros(N, dtype=np.bool)
            mask[choices] = 1
            mask = torch.tensor(mask).view(N).to(coords.device)

            selected_coords = []
            for i in range(C + 3):
                selected_coord = torch.masked_select(coords[:, i], mask)
                selected_coords.append(selected_coord)
            selected_coords = torch.stack(selected_coords, dim=1)
            coords = selected_coords
        
        save_place = os.path.join(save_path, scene_id + '_vert.npy')
        np.save(save_place, coords)
        
        if visualize_path != None:
            if not os.path.exists(os.path.join(visualize_path, scene_id)):
                os.makedirs(os.path.join(visualize_path, scene_id))
            visualize_place = os.path.join(visualize_path, scene_id, scene_id + '_points.ply')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords[:, 0:3])
            o3d.io.write_point_cloud(visualize_place, pcd)
        print('Saved', scene_id)


