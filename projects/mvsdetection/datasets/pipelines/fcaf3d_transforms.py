import numpy as np 
import torch
from mmdet.datasets.builder import PIPELINES
from projects.mvsdetection.datasets.pipelines.atlas_transforms import transform_space


class TransformFeaturesBBoxes(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        flip_ratio_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 n_points=None,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 flip_ratio_horizontal=0.0,
                 flip_ratio_vertical=0.0):
        self.n_points = n_points
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        
        if flip_ratio_horizontal is not None:
            assert isinstance(
                flip_ratio_horizontal,
                (int, float)) and 0 <= flip_ratio_horizontal <= 1
        if flip_ratio_vertical is not None:
            assert isinstance(
                flip_ratio_vertical,
                (int, float)) and 0 <= flip_ratio_vertical <= 1
        self.flip_ratio_horizontal = flip_ratio_horizontal
        self.flip_ratio_vertical = flip_ratio_vertical

    def sample_points(self, points, num_samples):
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points [torch.Tensor], [N * C]: the points
            num_samples (int): Number of samples to be sampled.

        """
        replace = (points.shape[0] < num_samples)
        choices = np.random.choice(points.shape[0], num_samples, replace=replace)
        return points[choices]
        

    def translate(self, points, gt_bboxes):
        """Translate bounding boxes and points.

        Args:
            points [torch.Tensor], [N * C]: the points
            rotation [float]: Rotation angle.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        points = translate_points(points, trans_factor)
        gt_bboxes.translate(trans_factor)
        return points, gt_bboxes

    def rotate(self, points, gt_bboxes):
        """Rotate bounding boxes and points.

        Args:
            points [torch.Tensor], [N * C]: the points
            rotation [float]: Rotation angle.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])
        points = rotate_points(points, noise_rotation)
        gt_bboxes.rotate(noise_rotation)
        return points, gt_bboxes


    def scale(self, points, gt_bboxes):
        """Scale bounding boxes and points.

        Args:
            points [torch.Tensor], [N * C]: the points
            rotation [float]: Rotation angle.
        """
        scale = np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
        points = scale_points(points, scale)
        gt_bboxes.scale(scale)
        return points, gt_bboxes


    def flip(self, points, gt_bboxes, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        points = flip_points(points, direction)
        gt_bboxes.flip(direction)
        return points, gt_bboxes

    def __call__(self, points, gt_bboxes):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            points [torch array], [N * C]: [The feature points]
            gt_bboxes [DepthInstanceBoxes]: [the ground truth bounding boxes]
        """
        if self.n_points != None:
            points = self.sample_points(points, self.n_points)
        
        flip_horizontal = True if np.random.rand() < self.flip_ratio_horizontal else False
        flip_vertical = True if np.random.rand() < self.flip_ratio_vertical else False
        if flip_horizontal:
            points, gt_bboxes = self.flip(points, gt_bboxes, 'horizontal')
        if flip_vertical:
            points, gt_bboxes = self.flip(points, gt_bboxes, 'vertical')
        points, gt_bboxes = self.rotate(points, gt_bboxes)
        points, gt_bboxes = self.scale(points, gt_bboxes)
        points, gt_bboxes = self.translate(points, gt_bboxes)         
        return points, gt_bboxes




def rotate_points(points, rotation):
    """Rotate points with the given rotation matrix or angle.

    Args:
        points [torch.Tensor], [N * C]: the points
        rotation [float]: Rotation angle.
    """
    rotation = torch.tensor(rotation)
    rot_sin = torch.sin(rotation)
    rot_cos = torch.cos(rotation)
    rot_mat_T = torch.tensor([[rot_cos, -rot_sin, 0],
                              [rot_sin, rot_cos, 0],
                              [0, 0, 1]])
    rot_mat_T = rot_mat_T.T
    rot_mat_T = rot_mat_T.to(points.device)
    points[:, :3] = points[:, :3] @ rot_mat_T
    return points

def flip_points(points, direction='horizontal'):
    """Flip the points along given direction."""
    if direction == 'horizontal':
        points[:, 0] = -points[:, 0]
    elif direction == 'vertical':
        points[:, 1] = -points[:, 1]
    return points

def translate_points(points, trans_vector):
    """Translate points with the given translation vector.

    Args:
        points [torch.Tensor], [N * C]: the points
        trans_vector (np.ndarray, torch.Tensor): Translation vector of size 3
    """
    if not isinstance(trans_vector, torch.Tensor):
        trans_vector = torch.tensor(trans_vector)
    trans_vector = trans_vector.to(points.device)
    points[:, :3] = points[:, :3] + trans_vector
    return points

   
def scale_points(points, scale_factor):
    """Scale the points with horizontal and vertical scaling factors.

    Args:
        points [torch.Tensor], [N * C]: the points
        scale_factors (float): Scale factors to scale the points.
    """
    points[:, :3] = points[:, :3] * scale_factor
    return points



@PIPELINES.register_module()
class AtlasTransformSpaceDetection(object):
    def __init__(self, voxel_dim, origin=[0, 0, 0], test=False, mode='middle'):
        """
        Args:
            voxel_dim: tuple of 3 ints (nx,ny,nz) specifying 
                the size of the output volume
            origin: origin of the voxel volume after transformation(xyz position of voxel (0,0,0))
            test: whether the mode is train or test, if train, the bounding box will be changed, 
                  the reconstructed mesh's off will be this origin. if test, the bounding box will
                  not be changed, and the offset will be used to recover the results
            mode: 'middle' and 'origin', if middle, use the center of the original tsdf as the center 
                  of the model. if origin, use the origin of the original tsdf as the origin 
        """
        self.voxel_dim = voxel_dim
        self.origin = origin
        self.test = test 
        self.mode = mode

    def __call__(self, data):
        tsdf = data['tsdf_dict']['tsdf_gt_004']

        if self.mode == 'middle':
            # get corners of bounding volume
            voxel_dim = torch.tensor(tsdf.tsdf_vol.shape) * tsdf.voxel_size
            xmin, ymin, zmin = tsdf.origin[0]
            xmax, ymax, zmax = tsdf.origin[0] + voxel_dim
            corners2d = torch.tensor([[xmin, xmin, xmax, xmax],
                                  [ymin, ymax, ymin, ymax]], dtype=torch.float32)

            # get new bounding volume (add padding for data augmentation)
            xmin = corners2d[0].min()
            xmax = corners2d[0].max()
            ymin = corners2d[1].min()
            ymax = corners2d[1].max()
            zmin = zmin
            zmax = zmax

            start = torch.tensor([xmin, ymin, zmin])
            end = -torch.as_tensor(self.voxel_dim) * tsdf.voxel_size + torch.tensor([xmax, ymax, zmax])
            middle = start * 0.5 + end * 0.5 
            t = -middle
        elif self.mode == 'origin':
            voxel_size = tsdf.voxel_size 
            origin = tsdf.origin
            shift = torch.tensor([.5, .5, .5]) // voxel_size
            t = origin - shift * voxel_size
        else:
            raise NotImplementedError

        if self.test:
            data['offset'] = t
        else:
            data['offset'] = torch.tensor(self.origin, dtype=torch.float32)
            data['gt_bboxes_3d'].translate(t)
        
        
        T = torch.eye(4)
        T[:3,3] = t
        return transform_space(data, T.inverse(), self.voxel_dim, self.origin)

    def __repr__(self):
        return self.__class__.__name__
