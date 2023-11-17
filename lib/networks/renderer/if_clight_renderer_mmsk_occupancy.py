import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from . import if_clight_renderer_occupancy


class Renderer(if_clight_renderer_occupancy.Renderer):
    def __init__(self, net):
        super(Renderer, self).__init__(net)

    def prepare_inside_pts(self, pts, batch):

        sh = pts.shape
        pts = pts.view(sh[0], -1, sh[3])

        inside = torch.ones([pts.size(0),pts.size(1)]).bool()

        return inside

    def get_density_color(self, wpts, viewdir, inside, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)
        wpts = wpts[inside][None]
        viewdir = viewdir[inside][None]
        full_raw = torch.zeros([n_batch, n_pixel * n_sample, 4]).to(wpts)
        if inside.sum() == 0:
            return full_raw

        raw = raw_decoder(wpts, viewdir)
        full_raw[inside] = raw[0]

        return full_raw

    def get_density_color_layer(self, wpts, viewdir, inside, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)
        wpts = wpts[inside][None]
        viewdir = viewdir[inside][None]
        full_raw_smpl = torch.zeros([n_batch, n_pixel * n_sample, 4]).to(wpts)
        full_raw_cloth = torch.zeros([n_batch, n_pixel * n_sample, 4]).to(wpts)
        full_blending = torch.zeros([n_batch, n_pixel * n_sample, 3]).to(wpts)
        if inside.sum() == 0:
            return full_raw_smpl, full_raw_cloth
        
        raw_smpl, raw_cloth = raw_decoder(wpts, viewdir)
        full_raw_smpl[inside] = raw_smpl[0]
        full_raw_cloth[inside] = raw_cloth[0]

        return full_raw_smpl, full_raw_cloth
        
    def get_pixel_value(self, ray_o, ray_d, near, far,
                        sp_input, batch, coord_silhouette):
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)

        inside = self.prepare_inside_pts(wpts, batch)

        n_batch, n_pixel, n_sample = wpts.shape[:3]
        ptsdist = torch.cdist(wpts.view(n_batch,-1,3), self.deformedpersonsmpl, p=2)
        nndist = torch.squeeze(torch.min(ptsdist, 2)[0], -1)  # B*P
        ptsnearsurfacetag_smpl = torch.where(nndist > 0.02, torch.zeros_like(nndist), torch.ones_like(nndist))
        ptsdist = torch.cdist(wpts.view(n_batch,-1,3), self.deformedcloth, p=2)
        nndist = torch.squeeze(torch.min(ptsdist, 2)[0], -1)  # B*P
        ptsnearsurfacetag_cloth = torch.where(nndist > 0.02, torch.zeros_like(nndist), torch.ones_like(nndist))
        
        # viewing direction
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)

        raw_decoder = lambda x_point, viewdir_val: self.net.calculate_density_color_clothdeformation_layer(
            x_point, viewdir_val, sp_input)

        # compute the color and density
        wpts_raw_smpl, wpts_raw_cloth = self.get_density_color_layer(wpts, viewdir, inside, raw_decoder)

        # volume rendering for wpts
        n_batch, n_pixel, n_sample = wpts.shape[:3]

        wpts_blending = coord_silhouette.repeat(1,1,n_sample)
        wpts_blending = wpts_blending.reshape(-1, n_sample)
        raw_smpl = wpts_raw_smpl.reshape(-1, n_sample, 4)
        raw_cloth = wpts_raw_cloth.reshape(-1, n_sample, 4)

        raw_smpl[...,3] = raw_smpl[...,3]*ptsnearsurfacetag_smpl.reshape(-1, n_sample)
        raw_cloth[...,3] = raw_cloth[...,3]*ptsnearsurfacetag_cloth.reshape(-1, n_sample)

        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        
        rgb_map_full, depth_map_full, acc_map_full, weights_full, \
        rgb_map_s, depth_map_s, acc_map_s, weights_s, \
        rgb_map_d, depth_map_d, acc_map_d, weights_d, dynamicness_map = raw2outputs_blend(raw_smpl, raw_cloth, wpts_blending, z_vals, ray_d, cfg.raw_noise_std)
             
        ret = {
            'rgb_map': rgb_map_full.view(n_batch, n_pixel, -1),
            'rgb_map_s': rgb_map_s.view(n_batch, n_pixel, -1),
            'rgb_map_d': rgb_map_d.view(n_batch, n_pixel, -1),
            'acc_map': acc_map_full.view(n_batch, n_pixel),
            'weights': weights_full.view(n_batch, n_pixel, -1),
            'depth_map': depth_map_full.view(n_batch, n_pixel),
            'dynamicness_map': dynamicness_map.view(n_batch, n_pixel),
            'blending': coord_silhouette.view(n_batch, -1)
        }
            
        return ret
