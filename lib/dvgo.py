import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import intersect_sphere
import tinycudann as tcnn
import json


HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

class NeRF_bg_network(torch.nn.Module):
    def __init__(self,dim0_bg,rgbnet_bg_width,  rgbnet_bg_depth, encoding_dirs):

        super(NeRF_bg_network, self).__init__()
        # Set rgbnet background
        self.rgbnet_bg = nn.Sequential(
                nn.Linear(dim0_bg, rgbnet_bg_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_bg_width, rgbnet_bg_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_bg_depth-2)
                ]
            )

        self.directions_layers =  nn.Sequential(nn.Linear(rgbnet_bg_width + encoding_dirs, 3))

        nn.init.constant_(self.directions_layers[-1].bias, 0)
    
    def forward(self,points, dirs):

        output = self.rgbnet_bg(points)
        out_concat = torch.cat( [output,dirs ], -1     )
        final_out =  self.directions_layers(out_concat)

        return final_out
        


'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 nearest=False, pre_act_density=False, in_act_density=False,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 posbase_pe=5, viewbase_pe=4,
                 rgbnet_bg_depth=3, rgbnet_bg_width=128,
                 posbase_bg_pe=10, viewbase_bg_pe=4,rgbnet_bg_dim=12,world_scale_bg = None,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self.nearest = nearest
        self.pre_act_density = pre_act_density
        self.in_act_density = in_act_density
        if self.pre_act_density:
            print('dvgo: using pre_act_density may results in worse quality !!')
        if self.in_act_density:
            print('dvgo: using in_act_density may results in worse quality !!')


        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'posbase_pe': posbase_pe, 'viewbase_pe': viewbase_pe,
            'posbase_bg_pe': posbase_bg_pe, 'viewbase_bg_pe': viewbase_bg_pe,
            'rgbnet_bg_depth': rgbnet_bg_depth, 'rgbnet_bg_width': rgbnet_bg_width,'rgbnet_bg_dim':rgbnet_bg_dim,'world_scale_bg':world_scale_bg
        }
        self.rgbnet_full_implicit = rgbnet_full_implicit

        # Set background true
        self.bg_color = True
        if self.bg_color:
          # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_bg_dim = 0
            else:
                self.k0_bg_dim = rgbnet_bg_dim
            
            if world_scale_bg is None:
                self.world_size_bg = self.world_size.clone()*4
                self.rgbnet_kwargs['world_scale_bg'] = self.world_size_bg
            else:
                self.world_size_bg = world_scale_bg
            self.density_bg = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size_bg]))
            with open("base.json") as f:
	              config_hash = json.load(f)

            self.encoding_dir_bg = tcnn.Encoding(3, config_hash["dir_encoding"])
            self.encoding_bg = tcnn.Encoding(3, config_hash["encoding"])
            #self.network_hash = tcnn.Network(9, 3, config_hash["network"])


            #self.k0_bg = torch.nn.Parameter(torch.zeros([1, self.k0_bg_dim, *self.world_size_bg]))
            self.register_buffer('posfreq_bg', torch.FloatTensor([(2**i) for i in range(posbase_bg_pe)]))
            self.register_buffer('viewfreq_bg', torch.FloatTensor([(2**i) for i in range(viewbase_bg_pe)]))

            # Set dimensions from spatial position and viewdirs
            
            # Set rgbnet background
            self.rgbnet_bg = NeRF_bg_network(self.encoding_bg.n_output_dims ,rgbnet_bg_width,  rgbnet_bg_depth, self.encoding_dir_bg.n_output_dims)

        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))

            with open("base.json") as f:
	              config_hash = json.load(f)
            self.encoding_dir = tcnn.Encoding(3, config_hash["dir_encoding"])
            self.encoding = tcnn.Encoding(3, config_hash["encoding"])
            

            #self.k0_bg = torch.nn.Parameter(torch.zeros([1, self.k0_bg_dim, *self.world_size_bg]))
            self.register_buffer('posfreq_bg', torch.FloatTensor([(2**i) for i in range(posbase_bg_pe)]))
            self.register_buffer('viewfreq_bg', torch.FloatTensor([(2**i) for i in range(viewbase_bg_pe)]))

            # Set dimensions from spatial position and viewdirs
            dim0 = self.encoding.n_output_dims  + self.encoding_dir.n_output_dims #(4+4*posbase_bg_pe*2) + self.encoding_dir_bg.n_output_dims  # (4+4*posbase_bg_pe*2) +(3+3*viewbase_bg_pe*2)

            #dim0 = (3+3*posbase_pe*2) +  self.encoding_dir.n_output_dims # self.encoding_dir.n_output_dims # (3+3*viewbase_pe*2)
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('dvgo: feature voxel grid', self.k0.shape)
            print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_path is not None and mask_cache_path:
            self.mask_cache = MaskCache(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self._set_nonempty_mask()
        else:
            self.mask_cache = None
            self.nonempty_mask = None

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'fast_color_thres': self.fast_color_thres,
            **self.rgbnet_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None,None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        self.density[~self.nonempty_mask] = -100

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density[nearest_dist[None,None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.density).requires_grad_()
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                
                far_fg = (intersect_sphere(rays_o, rays_d)) # For debugging purposes, we will just get the maximum foreground
        
                near_fg = (near*torch.ones(far_fg.shape))

                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near_fg, max=far_fg)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near_fg, max=far_fg)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation(self):
        tv = total_variation(self.activate_density(self.density, 1), self.nonempty_mask)
        return tv

    def k0_total_variation(self):
        if self.rgbnet is not None:
            v = self.k0
        else:
            v = torch.sigmoid(self.k0)
        return total_variation(v, self.nonempty_mask)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
            for grid in grids
        ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def to_spherical(self, xyz_1_r):
      '''Function to transform from x,y,z coordinates to spherical coordinates'''
      # Get x,y and z positions
      x_pos = xyz_1_r[...,0]
      y_pos = xyz_1_r[...,1]
      z_pos = xyz_1_r[...,2]
      
      r_pos = xyz_1_r[...,3]+ 0.1  # Just in case, sum by a tiny number so we dont overflow
      #theta_pos= torch.acos(z_pos)/torch.pi # Perform some transformations to ensure the theta is between -0.5 and 0.4
      #phi_pos = torch.atan2(y_pos,x_pos)/(2*torch.pi) + 0.5  # Just sum by Pi so that our coordinates goes from -0.5 to 0.5

      sphc_coord = torch.stack([x_pos*r_pos,y_pos*r_pos, z_pos*r_pos],dim=-1)/1.1

      return sphc_coord

    def grid_sampler_bg(self, xyz_1_r, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operatio, reserved for the background operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'

        # Project to spherical coordinates
        xyz_prooj = self.to_spherical(xyz_1_r)
        
        shape = xyz_prooj.shape[:-1]
        xyz_prooj = xyz_prooj.reshape(1,1,1,-1,3)

        ind_norm = xyz_prooj.flip((-1,)) 
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
            for grid in grids
        ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst
    ######################################################################################
    # wrapper to simplify the use of nerfnet
    ######################################################################################
    def depth2pts_outside(self,ray_o, ray_d, depth):
        '''
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        '''
        # note: d1 becomes negative if this mid point is behind camera
        d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
        p_mid = ray_o + d1.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)
        ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
        d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
        p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        phi = torch.asin(p_mid_norm)
        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                       torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                       rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        # now calculate conventional depth
        depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
        return pts, depth_real


    def intersect_sphere(self, ray_o, ray_d):
        '''
        ray_o, ray_d: [..., 3]
        compute the depth of the intersection point between this ray and unit sphere

        FROM NERF++ CODE
        '''
        # note: d1 becomes negative if this mid point is behind camera
        d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
        p = ray_o + d1.unsqueeze(-1) * ray_d
        # consider the case where the ray does not intersect the sphere
        ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
        p_norm_sq = torch.sum(p * p, dim=-1)
        if (p_norm_sq >= 1.).any():
            raise Exception('Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
        d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

        return d1 + d2

    def sample_bg_points(self, rays_o, rays_d, rays_pts, stepsize,  **render_kwargs):
        '''Sample query of background points on rays'''

        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1

        # Squash bg_rays so that we can perform the necessary computations
        bg_rays_o = torch.reshape(rays_o, (-1, 3))
        bg_rays_d = torch.reshape(rays_d, (-1, 3))

        bg_rays_o = bg_rays_o.unsqueeze(1).repeat(1,N_samples,1)
        bg_rays_d = bg_rays_d.unsqueeze(1).repeat(1,N_samples,1)
        
        # Sample the inverse depths so that we can hav the background depths
        bg_z_vals = torch.linspace(0., 1., N_samples).unsqueeze(0).repeat(bg_rays_d.shape[0], 1)
        rays_pts_bg, _ = self.depth2pts_outside(bg_rays_o, bg_rays_d, bg_z_vals)

        # Return to original shape
        shape_bg = rays_pts.shape[:-1] + (4,)# Get the shape for the 4D background representation
        rays_pts_bg = torch.reshape(rays_pts_bg, shape_bg )

        # Get the depth values and dists
        #bg_z_vals = torch.flip(bg_z_vals, dims=[-1,])
        #bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        #bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        #bg_dists = torch.reshape(bg_dists, rays_pts.shape[:-1]) # Perform reshape just in case

        #return rays_pts_bg, bg_dists
        return rays_pts_bg


    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays'''
        # 0. First, determine the far distance for the rays
        far_fg = (intersect_sphere(rays_o, rays_d)) # For debugging purposes, we will just get the maximum foreground
        
        near_fg = (near*torch.ones(far_fg.shape))

        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d) # This equation is just putting a offset so not to divide by 0
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near_fg, max=far_fg)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near_fg, max=far_fg)
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)
        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * self.voxel_size * rng
        interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[...,None] | ((self.xyz_min>rays_pts) | (rays_pts>self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering'''

        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # update mask for query points in known free space
        if self.mask_cache is not None:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))

        # query for alpha
        alpha = torch.zeros_like(rays_pts[...,0])
        if self.pre_act_density:
            # pre-activation
            alpha[~mask_outbbox] = self.grid_sampler(
                    rays_pts[~mask_outbbox], self.activate_density(self.density, interval))
        elif self.in_act_density:
            # in-activation
            density = self.grid_sampler(rays_pts[~mask_outbbox], F.softplus(self.density + self.act_shift))
            alpha[~mask_outbbox] = 1 - torch.exp(-density * interval)
        else:
            # post-activation
            density = self.grid_sampler(rays_pts[~mask_outbbox], self.density)
            alpha[~mask_outbbox] = self.activate_density(density, interval)

        # compute accumulated transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # query for color
        mask = (weights > self.fast_color_thres)
        k0 = torch.zeros(*weights.shape, self.k0_dim).to(weights)
        if not self.rgbnet_full_implicit:
            k0[mask] = self.grid_sampler(rays_pts[mask], self.k0)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[..., 3:]
                k0_diffuse = k0[..., :3]
            #viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            #viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = self.encoding_dir((torch.reshape(viewdirs,(-1,3)).unsqueeze(-1)+1)*.5 ) 
            viewdirs_emb = torch.reshape(viewdirs_emb, viewdirs.shape[:-1] + viewdirs_emb.shape[-1:] )

            rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
            xyz_emb = self.encoding(torch.reshape(rays_xyz,(-1,3)) ) 
            xyz_emb = torch.reshape(xyz_emb, rays_xyz.shape[:-1] + xyz_emb.shape[-1:] )
            #xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
            #xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)
            rgb_feat = torch.cat([
                k0_view[mask],
                xyz_emb,
                # TODO: use `rearrange' to make it readable
                viewdirs_emb.flatten(0,-2).unsqueeze(-2).repeat(1,weights.shape[-1],1)[mask.flatten(0,-2)]
            ], -1)
            rgb_logit = torch.zeros(*weights.shape, 3).to(weights)
            rgb_logit[mask] = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb_logit[mask] = rgb_logit[mask] + k0_diffuse
                rgb = torch.sigmoid(rgb_logit)

        # Ray marching

        if self.bg_color:

          rays_pts_bg = self.sample_bg_points(rays_o=rays_o,rays_d=rays_d,rays_pts=rays_pts, **render_kwargs)

          # In here, we create the density_bg and alpha_bg for background
          density_bg = self.grid_sampler_bg(rays_pts_bg,  self.density_bg)
          alpha_bg = self.activate_density(density_bg, None)

          weights_bg, alphainv_cum_bg = get_ray_marching_ray(alpha_bg )
          mask_bg = ( weights_bg > self.fast_color_thres   )

          #k0_bg = torch.zeros(*weights_bg.shape, self.k0_bg_dim).to(weights_bg)

          #k0_view_bg = k0_bg

          viewdirs_bg_emb = self.encoding_dir_bg((torch.reshape(viewdirs,(-1,3)).unsqueeze(-1) +1)*.5) 
          viewdirs_bg_emb = torch.reshape(viewdirs_bg_emb, viewdirs.shape[:-1] + viewdirs_bg_emb.shape[-1:] )

          #viewdirs_bg_emb_o = (viewdirs.unsqueeze(-1) * self.viewfreq_bg).flatten(-2)
          #viewdirs_bg_emb_o = torch.cat([viewdirs, viewdirs_bg_emb.sin(), viewdirs_bg_emb.cos()], -1)

          rays_xyz_bg =  self.to_spherical(rays_pts_bg[mask_bg])
          xyz_emb_bg = self.encoding_bg((torch.reshape(rays_xyz_bg,(-1,3)) + 1)/2) 
          xyz_emb_bg = torch.reshape(xyz_emb_bg, rays_xyz_bg.shape[:-1] + xyz_emb_bg.shape[-1:] )

          #xyz_emb_bg = (rays_xyz_bg.unsqueeze(-1) * self.posfreq_bg).flatten(-2)
          #xyz_emb_bg = torch.cat([rays_xyz_bg, xyz_emb_bg.sin(), xyz_emb_bg.cos()], -1)

          #rgb_bg_feat = torch.cat([
          #      #k0_view_bg[mask_bg],
          #      xyz_emb_bg,
          #      # TODO: use `rearrange' to make it readable
          #      viewdirs_bg_emb.flatten(0,-2).unsqueeze(-2).repeat(1,weights_bg.shape[-1],1)[mask_bg.flatten(0,-2)]
          #  ], -1)
          #
          rgb_bg_logit = torch.zeros(*weights_bg.shape, 3).to(weights_bg)
          rgb_bg_logit[mask_bg] = self.rgbnet_bg(xyz_emb_bg,viewdirs_bg_emb.flatten(0,-2).unsqueeze(-2).repeat(1,weights_bg.shape[-1],1)[mask_bg.flatten(0,-2)] )

          rgb_bg = torch.sigmoid(rgb_bg_logit)

          bg_color = (weights_bg[...,None] * rgb_bg).sum(-2) 

        else:
          bg_color = render_kwargs['bg']

        rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * bg_color
        rgb_marched = rgb_marched.clamp(0, 1)
        depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[...,-1]
        disp = 1 / depth
        
        #if self.bg_color:
        #  rgb = torch.cat([rgb, rgb_bg],dim=-2)
        #  weights = torch.cat([weights, weights_bg], dim=-1)


        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask,
        })
        return ret_dict


''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''
class MaskCache(nn.Module):
    def __init__(self, path, mask_cache_thres, ks=3):
        super().__init__()
        st = torch.load(path)
        self.mask_cache_thres = mask_cache_thres
        self.register_buffer('xyz_min', torch.FloatTensor(st['MaskCache_kwargs']['xyz_min']))
        self.register_buffer('xyz_max', torch.FloatTensor(st['MaskCache_kwargs']['xyz_max']))
        self.register_buffer('density', F.max_pool3d(
            st['model_state_dict']['density'], kernel_size=ks, padding=ks//2, stride=1))
        self.act_shift = st['MaskCache_kwargs']['act_shift']
        self.voxel_size_ratio = st['MaskCache_kwargs']['voxel_size_ratio']
        self.nearest = st['MaskCache_kwargs'].get('nearest', False)
        self.pre_act_density = st['MaskCache_kwargs'].get('pre_act_density', False)
        self.in_act_density = st['MaskCache_kwargs'].get('in_act_density', False)

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if self.nearest:
            density = F.grid_sample(self.density, ind_norm, align_corners=True, mode='nearest')
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        elif self.pre_act_density:
            alpha = 1 - torch.exp(-F.softplus(self.density + self.act_shift) * self.voxel_size_ratio)
            alpha = F.grid_sample(self.density, ind_norm, align_corners=True)
        elif self.in_act_density:
            density = F.grid_sample(F.softplus(self.density + self.act_shift), ind_norm, align_corners=True)
            alpha = 1 - torch.exp(-density * self.voxel_size_ratio)
        else:
            density = F.grid_sample(self.density, ind_norm, align_corners=True)
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        alpha = alpha.reshape(*shape)
        return (alpha >= self.mask_cache_thres)


''' Misc
'''
def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[...,[0]]), p.clamp_min(1e-10).cumprod(-1)], -1)

def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1-alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum

def total_variation(v, mask=None):
    tv2 = v.diff(dim=2).abs()
    tv3 = v.diff(dim=3).abs()
    tv4 = v.diff(dim=4).abs()
    if mask is not None:
        tv2 = tv2[mask[:,:,:-1] & mask[:,:,1:]]
        tv3 = tv3[mask[:,:,:,:-1] & mask[:,:,:,1:]]
        tv4 = tv4[mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            rays_pts, mask_outbbox = model.sample_ray(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs)
            mask_outbbox[~mask_outbbox] |= (~model.mask_cache(rays_pts[~mask_outbbox]))
            mask[i:i+CHUNK] &= (~mask_outbbox).any(-1).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

