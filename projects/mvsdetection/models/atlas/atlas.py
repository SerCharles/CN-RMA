import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict
from mmdet.models import DETECTORS 
from mmdet.models.builder import build_backbone, build_head, build_neck
from projects.mvsdetection.datasets.tsdf import TSDF, coordinates

@DETECTORS.register_module()
class Atlas(nn.Module):
    def __init__(self, pixel_mean, pixel_std, voxel_size, n_scales, voxel_dim_train, voxel_dim_test, origin, backbone2d_stride, 
                 backbone2d, feature_2d, backbone_3d, tsdf_head, 
                 train_cfg=None, test_cfg=None, pretrained=None):
        super(Atlas, self).__init__()
        # networks
        self.fpn = build_backbone(backbone2d)
        self.feature_2d = build_backbone(feature_2d)
        self.backbone3d = build_backbone(backbone_3d)
        self.tsdf_head = build_head(tsdf_head)

        # other hparams
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.voxel_size = voxel_size
        self.n_scales = n_scales 
        self.voxel_dim_train = voxel_dim_train
        self.voxel_dim_test = voxel_dim_test

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

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def backbone2d(self, image):
        x = self.fpn(image)
        x = self.feature_2d(x)
        return x

    def inference1(self, projection, image=None, feature=None):
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

    def inference2(self, targets=None):
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

        volume = self.volume / self.valid

        # remove nans (where self.valid==0)
        volume = volume.transpose(0,1)
        volume[:,self.valid.squeeze(1)==0]=0
        volume = volume.transpose(0,1)

        x = self.backbone3d(volume)
        return self.tsdf_head(x, targets)


    def forward_train(self, inputs):
        self.voxel_dim = self.voxel_dim_train
        self.initialize_volume()
        
        image = inputs['imgs']
        projection = inputs['projection']

        # get targets if they are in the batch
        targets3d = inputs['tsdf_list']
        targets3d = targets3d if targets3d else None


        # transpose batch and time so we can accumulate sequentially
        images = image.transpose(0,1)
        projections = projection.transpose(0,1)


        # run all images through 2d cnn together to share batchnorm stats
        image = images.reshape(images.shape[0]*images.shape[1], *images.shape[2:])
        image = self.normalizer(image)
        
        # CUDA MEMORY NOT ENOUGH
        '''
        image_size = int(image.shape[0] // 5)
        features = []
        for i in range(5):
            image_now = image[i * image_size : (i + 1) * image_size, :, :, :]
            feature_now = self.backbone2d(image_now)
            features.append(feature_now)
        features = torch.cat(features, dim=0)
        '''
        
        #original method
        features = self.backbone2d(image)
        
        
        
        # reshape back
        features = features.view(images.shape[0], images.shape[1], *features.shape[1:])

        for projection, feature in zip(projections, features):
            self.inference1(projection, feature=feature)

        # run 3d cnn
        outputs3d, losses3d = self.inference2(targets3d)
        
        
        '''
        results = self.post_process(outputs3d, inputs)
        #results = self.post_process({}, inputs)
        import os 
        save_path = '/data/shenguanlin/atlas/results'
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        for result in results:
            scene_id = result['scene']
            tsdf_pred = result['scene_tsdf']
            mesh_pred = tsdf_pred.get_mesh()
            if not os.path.exists(os.path.join(save_path, scene_id)):
                os.makedirs(os.path.join(save_path, scene_id))
            tsdf_pred.save(os.path.join(save_path, scene_id, scene_id + '.npz'))
            mesh_pred.export(os.path.join(save_path, scene_id, scene_id + '.ply'))
            kebab = result['kebab'].get_mesh()
            kebab.export(os.path.join(save_path, scene_id, scene_id + '_gt.ply'))
        '''
        
        
        return losses3d
    
    def forward_test(self, inputs):
        self.voxel_dim = self.voxel_dim_test
        self.initialize_volume()

        
        image = inputs['imgs']
        projection = inputs['projection']

        # get targets if they are in the batch
        targets3d = inputs['tsdf_list']
        targets3d = targets3d if targets3d else None
        
        # transpose batch and time so we can accumulate sequentially
        images = image.transpose(0,1)
        projections = projection.transpose(0,1)
        
        
        for projection, image in zip(projections, images):
            self.inference1(projection, image=image)
        
        
        '''
        # run all images through 2d cnn together to share batchnorm stats
        image = images.reshape(images.shape[0]*images.shape[1], *images.shape[2:])
        image = self.normalizer(image)
        features = self.backbone2d(image)
        
        # reshape back
        features = features.view(images.shape[0], images.shape[1], *features.shape[1:])

        for projection, feature in zip(projections, features):
            self.inference1(projection, feature=feature)
        '''
        
        # run 3d cnn
        outputs3d, losses3d = self.inference2(targets3d)
        print(losses3d)
        
        results = self.post_process(outputs3d, inputs)
        
        import os 
        save_path = '/data/shenguanlin/atlas/results'
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        for result in results:
            scene_id = result['scene']
            tsdf_pred = result['scene_tsdf']
            mesh_pred = tsdf_pred.get_mesh()
            if not os.path.exists(os.path.join(save_path, scene_id)):
                os.makedirs(os.path.join(save_path, scene_id))
            tsdf_pred.save(os.path.join(save_path, scene_id, scene_id + '.npz'))
            mesh_pred.export(os.path.join(save_path, scene_id, scene_id + '.ply'))
            #kebab = result['kebab'].get_mesh()
            #kebab.export(os.path.join(save_path, scene_id, scene_id + '_gt.ply'))
        
        return results

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
            
            #kebab = TSDF(self.voxel_size, self.origin, inputs['tsdf_list']['tsdf_gt_004'][i].squeeze(0))
            #kebab.origin = offset
            #out['kebab'] = kebab
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
        return data 
        
    
    def init_weights(self):
        pass


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
