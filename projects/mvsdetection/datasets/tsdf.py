# Modified from
# https://github.com/magicleap/Atlas/blob/master/atlas/tsdf.py
# Copyright (c) MagicLeap, Inc. and its affiliates.
"""TSDFFusion and TSDF classes
"""

import numpy as np
from skimage import measure
import torch
import trimesh



def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))

class TSDF():
    """ class to hold a truncated signed distance function (TSDF)

    Holds the TSDF volume along with meta data like voxel size and origin
    required to interpret the tsdf tensor.
    Also implements basic opperations on a TSDF like extracting a mesh.

    """

    def __init__(self, voxel_size, origin, tsdf_vol):
        """
        Args:
            voxel_size: metric size of voxels (ex: .04m)
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
            tsdf_vol: tensor of size hxwxd containing the TSDF values
        """
        self.voxel_size = voxel_size
        self.origin = origin
        self.tsdf_vol = tsdf_vol
        self.device = tsdf_vol.device

    def save(self, fname):
        data = {'origin': self.origin.cpu().numpy(),
                'voxel_size': self.voxel_size,
                'tsdf': self.tsdf_vol.detach().cpu().numpy()}
        np.savez_compressed(fname, **data)

    @classmethod
    def load(cls, fname):
        """ Load a tsdf from disk (stored as npz).

        Args:
            fname: path to archive
        Returns:
            TSDF
        """
        with np.load(fname) as data:
            voxel_size = data['voxel_size'].item()
            origin = torch.as_tensor(data['origin']).view(1,3)
            tsdf_vol = torch.as_tensor(data['tsdf'])
            ret = cls(voxel_size, origin, tsdf_vol)
        return ret

    def to(self, device):
        """ Move tensors to a device"""
        self.origin = self.origin.to(device)
        self.tsdf_vol = self.tsdf_vol.to(device)
        self.device = device
        return self

    def get_mesh(self):
        """ Extract a mesh from the TSDF using marching cubes

        If TSDF also has atribute_vols, these are extracted as
        vertex_attributes. The mesh is also colored using the cmap 

        Args:
            attribute: which tsdf attribute is used to color the mesh
            cmap: colormap for converting the attribute to a color

        Returns:
            trimesh.Trimesh
        """

        tsdf_vol = self.tsdf_vol.detach().clone()

        # measure.marching_cubes() likes positive 
        # values in front of surface
        tsdf_vol = -tsdf_vol

        # don't close surfaces using unknown-empty boundry
        tsdf_vol[tsdf_vol==-1]=1

        tsdf_vol = tsdf_vol.clamp(-1,1).cpu().numpy()

        if tsdf_vol.min()>=0 or tsdf_vol.max()<=0:
            return trimesh.Trimesh(vertices=np.zeros((0,3)))

        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)

        verts = verts * self.voxel_size + self.origin.cpu().numpy()

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
        return mesh


    def transform(self, transform=None, voxel_dim=None, origin=None,
                  align_corners=False):
        """ Applies a 3x4 linear transformation to the TSDF.

        Each voxel is moved according to the transformation and a new volume
        is constructed with the result.

        Args:
            transform: 3x4 linear transform
            voxel_dim: size of output voxel volume to construct (nx,ny,nz)
                default (None) is the same size as the input
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))0
                default (None) is the same as the input
        
        Returns:
            A new TSDF with the transformed coordinates
        """

        device = self.tsdf_vol.device

        old_voxel_dim = list(self.tsdf_vol.size())
        old_origin = self.origin

        if transform is None:
            transform = torch.eye(4, device=device)
        if voxel_dim is None:
            voxel_dim = old_voxel_dim
        if origin is None:
            origin = old_origin
        else:
            origin = torch.tensor(origin, dtype=torch.float, device=device).view(1,3)

        coords = coordinates(voxel_dim, device)
        world = coords.type(torch.float) * self.voxel_size + origin.T
        world = torch.cat((world, torch.ones_like(world[:1]) ), dim=0)
        world = transform[:3,:] @ world
        coords = (world - old_origin.T) / self.voxel_size

        # grid sample expects coords in [-1,1]
        coords = 2*coords/(torch.tensor(old_voxel_dim, device=device)-1).view(3,1)-1
        coords = coords[[2,1,0]].T.view([1]+voxel_dim+[3])

        # bilinear interpolation near surface,
        # no interpolation along -1,1 boundry
        tsdf_vol = torch.nn.functional.grid_sample(
            self.tsdf_vol.view([1,1]+old_voxel_dim),
            coords, mode='nearest', align_corners=align_corners
        ).squeeze()
        tsdf_vol_bilin = torch.nn.functional.grid_sample(
            self.tsdf_vol.view([1,1]+old_voxel_dim), coords, mode='bilinear',
            align_corners=align_corners
        ).squeeze()
        mask = tsdf_vol.abs()<1
        tsdf_vol[mask] = tsdf_vol_bilin[mask]

        # padding_mode='ones' does not exist for grid_sample so replace 
        # elements that were on the boarder with 1.
        # voxels beyond full volume (prior to croping) should be marked as empty
        mask = (coords.abs()>=1).squeeze(0).any(3)
        tsdf_vol[mask] = 1

        return TSDF(self.voxel_size, origin, tsdf_vol)


