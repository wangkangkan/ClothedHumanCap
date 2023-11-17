import torch
from lib.config import cfg
from .nerf_net_utils import *
import os
import numpy as np
import trimesh

from pytorch3d.renderer.cameras import PerspectiveCameras
import pytorch3d.structures as struct
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.structures import Pointclouds

class Renderer:
    def __init__(self, net):
        self.net = net

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        faces_path = os.path.join(cfg.train_dataset.data_root, 'templatedeform/modeltri.txt')
        faces = np.loadtxt(faces_path)-1
        self.faces = torch.LongTensor(faces).to(self.device)
        self.faces = self.faces[None, :, :]
        
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeform/smpltri.txt')) - 1
        self.smplfaces = torch.LongTensor(npfaces).to(self.device)
        self.smplfaces = self.smplfaces[None, :, :]
        
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/desmpltri.txt')) - 1
        self.desmplfaces = torch.LongTensor(npfaces).to(self.device)
        self.desmplfaces = self.desmplfaces[None, :, :]

        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/desmpl_vfidx.txt')) - 1        
        self.desmpl_vfidx = torch.LongTensor(npfaces).to(self.device)
        
        templateshape_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/templateshapeT.txt')
        templateshape = np.loadtxt(templateshape_path)
        templateshape = templateshape
        self.templateshape = torch.Tensor(templateshape).to(self.device)
        self.templateshape = self.templateshape[None, :, :]
        
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_face.txt')) - 1
        self.clothfaces = torch.LongTensor(npfaces).to(self.device)
        self.clothfaces = self.clothfaces[None, :, :]
        
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_watertight_face.txt')) - 1
        self.clothes_watertight_face = torch.LongTensor(npfaces).to(self.device)
        self.clothes_watertight_face = self.clothes_watertight_face[None, :, :]
        
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/cloth_vfidx.txt')) - 1
        self.cloth_vfidx = torch.LongTensor(npfaces).to(self.device)
        templatesmpl_path = os.path.join(cfg.train_dataset.data_root,
                                         'templatedeformT/vpersonalshape.txt')  # smpldeform/vpersonalshape
        templatesmpl = np.loadtxt(templatesmpl_path)
        templatesmpl = templatesmpl
        templatesmpl = torch.Tensor(templatesmpl).to(self.device)
        # loading template deformation graph
        templateshape_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_vert.txt')
        templatecloth = np.loadtxt(templateshape_path)
        templatecloth = torch.Tensor(templatecloth).to(self.device)
        cloth_ptsdist = torch.cdist(templatecloth[None,...], templatesmpl[None,...], p=2)
        cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        self.cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P
        
        self.meshface = torch.cat([self.clothfaces, self.desmplfaces+templatecloth.shape[0]], dim=1)
        
        self.l1loss = torch.nn.L1Loss()
        
        attachidx = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/attachidx.txt')) - 1
        attachidx = torch.LongTensor(attachidx).to(self.device)
        self.attachtag = torch.zeros_like(templatecloth[:,0])
        self.attachtag[attachidx] = 1

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape # [B,N,2], x,y
        idx = [torch.full([sh[1]], i, dtype=torch.long) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1) 

        out_sh, _ = torch.max(batch['out_sh'], dim=0) 
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        # used for feature interpolation
        sp_input['bounds'] = batch['bounds']
        sp_input['R'] = batch['R']
        sp_input['Th'] = batch['Th']

        # used for color function
        sp_input['latent_index'] = batch['latent_index']
        sp_input['frame_index'] = batch['frame_index']

        sp_input['vert'] = batch['vert']
        sp_input['A'] = batch['A']
        sp_input['RT'] = batch['RT']
        sp_input['K'] = batch['K']
        sp_input['msk'] = batch['msk']
        sp_input['meshpts_cloth'] = batch['meshpts_cloth']
        sp_input['sdf_cloth'] = batch['sdf_cloth']
        sp_input['normal_cloth'] = batch['normal_cloth']
        sp_input['tpose_cloth'] = batch['tpose_cloth']
        
        sp_input['smplpose'] = batch['smplpose']
        sp_input['smplshape'] = batch['smplshape']
        return sp_input

    def samplingpoint_learningmeshsdf(self, batch, templateclothverts):
    
        clothfaces = self.clothes_watertight_face.squeeze(0)
        clothverts = templateclothverts
                
        clothfacevert1 = clothverts[clothfaces[:,0],:] 
        clothfacevert2 = clothverts[clothfaces[:,1],:]      
        clothfacevert3 = clothverts[clothfaces[:,2],:] 
        
        n_sample = clothfaces.shape[0]
        faceweight = torch.rand(n_sample, 3)
        faceweight = faceweight.to(self.device)
        faceweightsum = torch.sum(faceweight,1)
        faceweight = faceweight/faceweightsum[:,None]
        meshpts_cloth = clothfacevert1*faceweight[:,0][:,None]+clothfacevert2*faceweight[:,1][:,None]+clothfacevert3*faceweight[:,2][:,None]

        _,trinormal = mesh_face_areas_normals(clothverts.view(-1,3), self.clothes_watertight_face.view(-1,3))
        sdf_cloth = torch.zeros([n_sample]).to(self.device)
        normal_cloth = trinormal
        
        min_xyz, _ = torch.min(clothverts, axis=0)
        max_xyz, _ = torch.max(clothverts, axis=0)
        min_xyz = min_xyz - 0.05
        max_xyz = max_xyz + 0.05
       
        gridsample = n_sample//2       
        vals = torch.rand(gridsample, 3).to(self.device)
        samplepts = (max_xyz - min_xyz) * vals + min_xyz
        
        batch['meshpts_cloth'] = meshpts_cloth[None,...]
        batch['sdf_cloth'] = sdf_cloth[None,...]
        batch['normal_cloth'] = normal_cloth[None,...]
        batch['tpose_cloth'] = samplepts[None,...]
        
        return batch

    def get_density_color_layer(self, wpts, viewdir, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)
        raw_smpl, raw_cloth = raw_decoder(wpts, viewdir)

        return raw_smpl, raw_cloth#, blending
          
    def get_pixel_value(self, ray_o, ray_d, near, far,
                        sp_input, batch, coord_silhouette):
        # sampling points along camera rays, geometryzero
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)

        n_batch, n_pixel, n_sample = wpts.shape[:3]
        
        ptsdist = torch.cdist(wpts.view(n_batch,-1,3), self.deformedpersonsmpl, p=2)
        nndist = torch.squeeze(torch.min(ptsdist, 2)[0], -1)  # B*P
        ptsnearsurfacetag_smpl = torch.where(nndist > 0.02, torch.zeros_like(nndist), torch.ones_like(nndist))
        ptsdist = torch.cdist(wpts.view(n_batch,-1,3), self.deformedcloth, p=2)
        nndist = torch.squeeze(torch.min(ptsdist, 2)[0], -1)  # B*P
        ptsnearsurfacetag_cloth = torch.where(nndist > 0.02, torch.zeros_like(nndist), torch.ones_like(nndist))

        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
        
        raw_decoder = lambda x_point, viewdir_val: self.net.calculate_density_color_clothdeformation_layer(
            x_point, viewdir_val, sp_input)

        wpts_raw_smpl, wpts_raw_cloth = self.get_density_color_layer(wpts, viewdir, raw_decoder)

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
        rgb_map_d, depth_map_d, acc_map_d, weights_d, dynamicness_map = raw2outputs_blend(raw_smpl, 
                                                                                          raw_cloth, 
                                                                                          wpts_blending, 
                                                                                          z_vals, 
                                                                                          ray_d, 
                                                                                          cfg.raw_noise_std)
                
        ret = {
            'rgb_map': rgb_map_full.view(n_batch, n_pixel, -1),
            'acc_map': acc_map_full.view(n_batch, n_pixel),
            'weights': weights_full.view(n_batch, n_pixel, -1),
            'depth_map': depth_map_full.view(n_batch, n_pixel),
            'rgb_map_s': rgb_map_s.view(n_batch, n_pixel, -1),
            'rgb_map_d': rgb_map_d.view(n_batch, n_pixel, -1)
        }

        return ret

    def computeinterpenetrationloss(self, graphdeformedverts):

        batch_size = graphdeformedverts.size(0)
        cloth_ptsdist = torch.cdist(graphdeformedverts, self.net.deformation_network.rawtemplatesmpl[None,...], p=2)
        cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P

        ptsnum = graphdeformedverts.size(1)
        templatevtnum = self.net.deformation_network.rawtemplatesmpl.size(0)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(batch_size)]
        idx = torch.cat(idx).to(self.device)
        bwidx = cloth_nnvidx.view(-1) + idx

        canpersonsmpl = self.net.deformation_network.rawtemplatesmpl
        selectpersonsmpl = canpersonsmpl[bwidx.long(), :]
        selectpersonsmpl_norm = self.net.deformation_network.smpl_vertnorm[bwidx.long(), :]
              
        normaldist = (selectpersonsmpl - graphdeformedverts.view(-1,3))*selectpersonsmpl_norm
 
        interpenetration = torch.sum(normaldist,1)
        interpenetrationloss = torch.mean(F.relu(interpenetration),0)
        
        return interpenetrationloss
     
    def computeinterpenetrationloss_posedsmpl(self, graphdeformedverts):

        batch_size = graphdeformedverts.size(0)
        cloth_ptsdist = torch.cdist(graphdeformedverts, self.net.deformation_network.smplgraphdeformedverts, p=2)
        cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P
        
        ptsnum = graphdeformedverts.size(1)
        templatevtnum = self.deformedpersonsmpl.size(1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(batch_size)]
        idx = torch.cat(idx).to(self.device)
        bwidx = cloth_nnvidx.view(-1) + idx
        canpersonsmpl = self.net.deformation_network.smplgraphdeformedverts.view(-1,3)
        selectpersonsmpl = canpersonsmpl[bwidx.long(), :]
        selectpersonsmpl_norm = self.net.deformation_network.deformedsmpl_vertnorm[bwidx.long(), :]
              
        normaldist = (selectpersonsmpl - graphdeformedverts.view(-1,3))*selectpersonsmpl_norm
 
        interpenetration = torch.sum(normaldist,1)
        interpenetrationloss = torch.mean(F.relu(interpenetration),0)
        
        _,trinormal = mesh_face_areas_normals(self.deformedpersonsmpl.view(-1,3), self.desmplfaces[0])
         
        deformedposedsmpl_vertnorm = trinormal[self.desmpl_vfidx,:]

        pospersonsmpl = self.deformedpersonsmpl.view(-1,3)
        selectpospersonsmpl = pospersonsmpl[bwidx.long(), :]
        selectpospersonsmpl_norm = deformedposedsmpl_vertnorm[bwidx.long(), :]
              
        normaldist1 = (selectpospersonsmpl - self.deformedcloth.view(-1,3))*selectpospersonsmpl_norm
 
        interpenetration1 = torch.sum(normaldist1,1)
        interpenetrationloss1 = torch.mean(F.relu(interpenetration1),0)
        
        return interpenetrationloss+interpenetrationloss1

    def render_deformation(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape

        sp_input = self.prepare_sp_input(batch)

        # Predict Deformation
        latent_index = sp_input['latent_index'].item()
        if latent_index>=150:
            sp_input['latent_index'] = sp_input['latent_index'] - 150
            self.deformation_affine, self.deformation_transl = \
                self.net.deformation_network2.predicting_deformation(sp_input)
                
            self.deformation_affine_smpl, self.deformation_transl_smpl = \
                self.net.deformation_network2.predicting_deformation_smpl(sp_input)
            sp_input['latent_index'] = sp_input['latent_index'] + 150
        else:
            self.deformation_affine, self.deformation_transl = \
                self.net.deformation_network1.predicting_deformation(sp_input)
                
            self.deformation_affine_smpl, self.deformation_transl_smpl = \
                self.net.deformation_network1.predicting_deformation_smpl(sp_input)
                
        self.net.deformation_network.deformation_affine = self.deformation_affine
        self.net.deformation_network.deformation_transl = self.deformation_transl
        self.net.deformation_network.deformation_affine_smpl = self.deformation_affine_smpl
        self.net.deformation_network.deformation_transl_smpl = self.deformation_transl_smpl
        
        # Get cloth params   
        tempclothpara = self.net.tempclothpara(torch.zeros(1).to(torch.int64).to(self.device))
        tempclothpara = 2*torch.sigmoid(tempclothpara)-1 
        zeropara = torch.zeros_like(tempclothpara).to(self.device)
        tempclothpara = torch.cat((tempclothpara,zeropara),-1)
        
        # Cloth Simulation
        simulatedcanoncloth = self.net.cloth_simulation(sp_input['smplpose'],sp_input['smplshape'],tempclothpara)
        self.simulatedcloth =self.net.deformation_network.LBS_simulatedcloth(simulatedcanoncloth, sp_input)
        
        # Zero-Pose Cloth vert
        clothvert = self.net.cloth_simulation(torch.zeros_like(sp_input['smplpose']).to(self.device),sp_input['smplshape'],tempclothpara)

        self.net.deformation_network.update_embeddedgraph(clothvert.reshape(-1,3))

        # Compute Smooth Loss
        self.smoothloss = self.net.deformation_network.deformationsmoothloss()
        self.smoothloss_smpl = self.net.deformation_network.deformationsmoothloss_smpl()

        # Deform cloth and SMPL
        self.deformedpersonsmpl, smplgraphdeformedverts = self.net.deformation_network.deformingsmpl_graphdeform_LBS(sp_input)
        
        self.deformedcloth, graphdeformedverts = self.net.deformation_network.deformingcloth_graphdeform_LBS(sp_input)
        
        # Compute Interpenetration Loss in canonical and posed space
        self.interpenetrationloss = self.computeinterpenetrationloss(simulatedcanoncloth)
        self.interploss_graphdeform = self.computeinterpenetrationloss_posedsmpl(graphdeformedverts)

        batch = self.samplingpoint_learningmeshsdf(batch, self.net.deformation_network.templateshape)
        self.graphdeform_loss = self.l1loss(graphdeformedverts, simulatedcanoncloth)
        
        # Compute Attach Loss
        ptsdist = torch.cdist(clothvert, self.net.deformation_network.templatesmpl[None,...], p=2)
        nndist = torch.squeeze(torch.min(ptsdist, 2)[0], -1)  # B*P
        
        nnidx = torch.squeeze(torch.min(ptsdist, 2)[1], -1).view(-1)
        defptsdist = torch.sqrt(torch.sum((self.deformedcloth-self.deformedpersonsmpl[:,nnidx,:])**2,-1))       
        self.attach_loss = (torch.sum((defptsdist-0.01)**2*self.attachtag)).mean()
                
        R, T = torch.split(batch['RT'], [3, 1], dim=-1)
        R = R.transpose(-1, -2)
        T = T.transpose(-1, -2)
        
        # Render mask on deformed mesh and the simulated mesh                              
        cameras = PerspectiveCameras(device='cuda',
                                     K=batch['pytorch3d_K'].float(),
                                     R=R.float(),
                                     T=T[0].float())
        
        meshvert = torch.cat([self.simulatedcloth, self.deformedpersonsmpl], dim=1)
        
        self.net.pcRender.rasterizer.cameras=cameras
        features=[torch.ones(meshvert.shape[1],1,device=self.device) for _ in range(1)]
        predicted_silhouette, frags = self.net.pcRender(Pointclouds(points=meshvert,features=features))
        predicted_silhouette = predicted_silhouette.squeeze(-1)
        self.IoUloss = ((predicted_silhouette - batch['msk']) ** 2).mean()
        
        meshvert_def = torch.cat([self.deformedcloth, self.deformedpersonsmpl], dim=1)
        
        self.net.pcRender_def.rasterizer.cameras=cameras
        features=[torch.ones(meshvert_def.shape[1],1,device=self.device) for _ in range(1)]
        predicted_silhouette_def,frags=self.net.pcRender_def(Pointclouds(points=meshvert_def,features=features))
        predicted_silhouette_def = predicted_silhouette_def.squeeze(-1)
        self.IoUloss_def = ((predicted_silhouette_def - batch['msk']) ** 2).mean()
        
        with torch.no_grad():
            frame_index = sp_input['frame_index'].item()
            # if frame_index == 0 or frame_index == 750 or frame_index == 780:
            if cfg.vis_mesh:
                npvertices = meshvert_def[0].detach().cpu().numpy()
                npfaces = self.meshface[0].detach().cpu().numpy()
                mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
                result_path = os.path.join(cfg.train_dataset.data_root,
                                           'deformedcansmpl/frmmodel/deformedverts{:04d}.obj'.format(frame_index))
                mesh.export(result_path)
        
        # Get Color    
        mesh = struct.Meshes(verts=meshvert_def, faces=self.meshface) 
        
        fragments = self.net.rasterizer(mesh, cameras=cameras)
        depth = fragments.zbuf
        face_idx_map = fragments.pix_to_face[..., 0]

        self.silhouette0 = face_idx_map>=self.clothfaces.shape[1]# body mask

        self.silhouette = ~self.silhouette0
       
        self.silhouette = self.silhouette.squeeze(-1).float()

        coord = batch['coord']#B*P*2
        batch_size = coord.size()[0]
        B = torch.arange(batch_size).view([batch_size, 1]).repeat(1, coord.size(1)).reshape(-1, 1)
        B = B.to(self.device)
        bcoord = torch.cat([B, coord.view(-1, 2)], dim=1)
        bcoord[:, 1] = 2 * torch.true_divide(bcoord[:, 1], cfg.H) - 1
        bcoord[:, 2] = 2 * torch.true_divide(bcoord[:, 2], cfg.W) - 1
        coordidx = bcoord[:, 1:3][:, [1, 0]].view(batch_size, -1, 2)[:, None]
        silhouette1 = self.silhouette[..., None].permute(0, 3, 1, 2)
        out = torch.nn.functional.grid_sample(silhouette1, coordidx, mode='nearest', padding_mode='border', align_corners=True)
        out = out.permute(0, 2, 3, 1)
        self.coord_silhouette = out.view(batch_size, -1)[...,None]  # out[:,0,0,:]
        clothrendermask = (face_idx_map<self.clothfaces.shape[1]) * (face_idx_map>-1)       
        self.clothrendermask = clothrendermask.squeeze(-1).float()  

        silhouette1 = self.clothrendermask[..., None].permute(0, 3, 1, 2)
        out = torch.nn.functional.grid_sample(silhouette1, coordidx, mode='nearest', padding_mode='border', align_corners=True)
        out = out.permute(0, 2, 3, 1)
        self.coordclothrendermask = out.view(batch_size, -1)[...,None]  # out[:,0,0,:]
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            coord_silhouette = self.coord_silhouette[:, i:i + chunk]

            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               sp_input, batch, coord_silhouette)
            ret_list.append(pixel_value)#, self.posedirs
        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret

