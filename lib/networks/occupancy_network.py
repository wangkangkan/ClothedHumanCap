import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from . import embedder
from lib.utils import net_utils
import os
import numpy as np

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRasterizer,
)
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor
)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sdf_network_cloth = SDFNetwork()
        self.color_network_smpl = ColorNetwork()
        self.color_network_cloth = ColorNetwork()
        
        self.occupancy_network_smpl = OccupancyNetwork()
        self.occupancy_network_cloth = OccupancyNetwork()
        
        self.deformation_network = VaryingClothDeformationNetwork()
        self.cloth_simulation = ClothSimulation()
        state_dict = torch.load('trained_model/cloth_simulation.pth.tar')
        self.cloth_simulation.model.load_state_dict(state_dict)
        
        net_utils.load_network(self, 'trained_model/occ/desmpl/', strict=False)
        
        pretrained_model = torch.load('trained_model/cloth/latest.pth')
        model_dict = self.occupancy_network_cloth.state_dict()        
        pretrained_dict = {k[24:]: v for k, v in pretrained_model['net'].items() if k[24:] in model_dict}        
        model_dict.update(pretrained_dict)
        self.occupancy_network_cloth.load_state_dict(model_dict)
        
        model_dict1 = self.sdf_network_cloth.state_dict()        
        pretrained_dict1 = {k[18:]: v for k, v in pretrained_model['net'].items() if k[18:] in model_dict1}        
        model_dict1.update(pretrained_dict1)
        self.sdf_network_cloth.load_state_dict(model_dict1)
        
        self.tempclothpara = nn.Embedding.from_pretrained(pretrained_model['net']['tempclothpara.weight'], freeze=True)
        
        cfg.num_train_frame = 150
        self.deformation_network1 = VaryingClothDeformationNetwork()
        self.deformation_network2 = VaryingClothDeformationNetwork()
        pretrained_model = torch.load('trained_model/def/latest1.pth')
        model_dict = self.deformation_network1.state_dict()        
        pretrained_dict = {k[20:]: v for k, v in pretrained_model['net'].items() if k[20:] in model_dict}        
        model_dict.update(pretrained_dict)
        self.deformation_network1.load_state_dict(model_dict)
        pretrained_model = torch.load('trained_model/def/latest2.pth')
        model_dict = self.deformation_network2.state_dict()        
        pretrained_dict = {k[20:]: v for k, v in pretrained_model['net'].items() if k[20:] in model_dict}        
        model_dict.update(pretrained_dict)
        self.deformation_network2.load_state_dict(model_dict)
        cfg.num_train_frame = 300

        templateshape_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_vert.txt')
        templateshape = np.loadtxt(templateshape_path)
        self.initclothvert = torch.Tensor(templateshape).to(self.device)
        
        elev = torch.linspace(0, 360, 4)
        azim = torch.linspace(-180, 180, 4)

        R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=(cfg.H, cfg.W),
            blur_radius=0,
            faces_per_pixel=1, 
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        
        raster_settings_silhouette = PointsRasterizationSettings(
            image_size=(cfg.H, cfg.W), 
            radius=0.005,
            bin_size=(92 if max(cfg.H, cfg.W)>1024 and max(cfg.H, cfg.W)<=2048 else None),
            points_per_pixel=50,
            )   
        self.pcRender=PointsRendererWithFrags(
            rasterizer=PointsRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
                compositor=AlphaCompositor(background_color=None)
            )#.to(self.device)
            
        self.pcRender_def=PointsRendererWithFrags(
            rasterizer=PointsRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
                compositor=AlphaCompositor(background_color=None)
            )#.to(self.device)    
                  
    def calculate_density_color_clothdeformation_layer(self, wpts, viewdir, sp_input):
        """
        calculate density and color
        """

        smpl_canpts, cloth_canpts = self.deformation_network.inversedeforming_samplepoints_layer(wpts, sp_input)#, posedirs
        smpllight_pts = embedder.xyz_embedder(smpl_canpts)
        clothlight_pts = embedder.xyz_embedder(cloth_canpts)
        
        alpha_smpl, _ = self.occupancy_network_smpl(smpllight_pts)
        
        alpha_cloth, _ = self.occupancy_network_cloth(clothlight_pts)

        rgb_smpl = self.color_network_smpl(smpllight_pts, viewdir, sp_input)
        rgb_cloth = self.color_network_cloth(clothlight_pts, viewdir, sp_input)
        

        raw_smpl = torch.cat([rgb_smpl, alpha_smpl], -1)
        raw_cloth = torch.cat([rgb_cloth, alpha_cloth], -1)

        return raw_smpl, raw_cloth
    
    def forward(self, sp_input, grid_coords, viewdir, light_pts):
        __import__('ipdb').set_trace()

class VaryingClothDeformationNetwork(nn.Module):
    def __init__(self):
        super(VaryingClothDeformationNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loading SMPL Template vertices
        templatesmpl_path = os.path.join(cfg.train_dataset.data_root,
                                         'templatedeformT/vpersonalshape.txt')  # smpldeform/vpersonalshape
        templatesmpl = np.loadtxt(templatesmpl_path)
        self.rawtemplatesmpl = torch.Tensor(templatesmpl).to(self.device)
        
        # Loading SMPL model faces
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeform/smpltri.txt')) - 1
        self.smplfaces = torch.LongTensor(npfaces).to(self.device)
        
        # Computing Face Normal
        _,trinormal = mesh_face_areas_normals(self.rawtemplatesmpl, self.smplfaces)
         
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/smpl_vfidx.txt')) - 1
        self.smpl_vfidx = torch.LongTensor(npfaces).to(self.device)
        
        # Get SMPL template vertices normals
        self.smpl_vertnorm = trinormal[self.smpl_vfidx,:]
        
        
        # loading cloth deformation graph node
        modelnodeidx_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/nodevidx.txt')
        modelnodeidx = np.loadtxt(modelnodeidx_path)
        self.modelnodeidx = torch.LongTensor(modelnodeidx).to(self.device)

        self.modelnodenum = self.modelnodeidx.size(0)
        modelnodeedge_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/modelnodeedge.txt')
        modelnodeedge = np.loadtxt(modelnodeedge_path) - 1
        self.modelnodeedge = torch.LongTensor(modelnodeedge).to(self.device)
        self.modelnodeedgenum = self.modelnodeedge.size(1)

        modelvertnode_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/modelvert_node.txt')
        modelvert_node = np.loadtxt(modelvertnode_path) - 1
        self.modelvert_node = torch.LongTensor(modelvert_node).to(self.device)
        modelvertnodeweight_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/modelvert_nodeweight.txt')
        modelvert_nodeweight = np.loadtxt(modelvertnodeweight_path)
        self.modelvert_nodeweight = torch.Tensor(modelvert_nodeweight).to(self.device)
        self.modelvert_nodenum = self.modelvert_node.size(1)


        hipidx_path = os.path.join(cfg.train_dataset.data_root,
                                   'templatedeformT/hipidx.txt')  # smpldeform/vpersonalshape
        hipidx = np.loadtxt(hipidx_path) - 1
        hipidx = torch.LongTensor(hipidx).to(self.device)
        skirtidx_path = os.path.join(cfg.train_dataset.data_root,
                                     'templatedeformT/cloth/skirtidx.txt')  # smpldeform/vpersonalshape
        skirtidx = np.loadtxt(skirtidx_path) - 1
        skirtidx = torch.LongTensor(skirtidx).to(self.device)

        
        # SMPL Skining Weight
        self.bw = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/skinweightnew.txt'))
        self.bw = torch.Tensor(self.bw)[None, ...].to(self.device)
 
        
        self.latentdeform = nn.Embedding(cfg.num_train_frame, 128)
        D = 8
        self.deformskips = [4]
        defW = 1024
        layers = [nn.Linear(128, defW)]  # node coding + latent code
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = defW
            if i in self.deformskips:
                in_channels += 128
            layers += [layer(in_channels, defW)]

        self.deformpara_linears = nn.ModuleList(layers)
        self.deformpara_finallinear = nn.Linear(defW, self.modelnodenum * 6)

        self.initsmplgraph()

    def initsmplgraph(self):
        
        # Loading deformed SMPL model
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/desmpl_vfidx.txt')) - 1
        self.desmpl_vfidx = torch.LongTensor(npfaces).to(self.device)
        
        templatesmpl_path = os.path.join(cfg.train_dataset.data_root,
                                         'templatedeformT/desmpl/desmplvt.txt')  # smpldeform/vpersonalshape
        templatesmpl = np.loadtxt(templatesmpl_path)
        self.templatesmpl = torch.Tensor(templatesmpl).to(self.device)
        self.smplvtnum = self.templatesmpl.size(0)
        
        smplnodeidx_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/nodevidx.txt')
        smplnodeidx = np.loadtxt(smplnodeidx_path)
        self.smplnodeidx = torch.LongTensor(smplnodeidx).to(self.device)
        self.smplnodepos = self.templatesmpl[self.smplnodeidx, :]
        
        self.smplnodenum = self.smplnodeidx.size(0)
        smplnodeedge_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/modelnodeedge.txt')
        smplnodeedge = np.loadtxt(smplnodeedge_path) - 1
        self.smplnodeedge = torch.LongTensor(smplnodeedge).to(self.device)
        self.smplnodeedgenum = self.smplnodeedge.size(1)
        
        smplvertnode_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/modelvert_node.txt')
        smplvert_node = np.loadtxt(smplvertnode_path) - 1
        self.smplvert_node = torch.LongTensor(smplvert_node).to(self.device)
        smplvertnodeweight_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/modelvert_nodeweight.txt')
        smplvert_nodeweight = np.loadtxt(smplvertnodeweight_path)
        self.smplvert_nodeweight = torch.Tensor(smplvert_nodeweight).to(self.device)
        self.smplvert_nodenum = self.smplvert_node.size(1)
        
           
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/desmpltri.txt')) - 1
        self.desmplfaces = torch.LongTensor(npfaces).to(self.device)
        
        bw = np.load(os.path.join(cfg.train_dataset.data_root, 'bw.npy'), allow_pickle=True)
        bw = torch.Tensor(bw).to(self.device)

        # get deformed SMPL skining weight
        ptsdist = torch.cdist(self.templatesmpl, self.rawtemplatesmpl, p=1)#self.templateshape
        minptsdist = torch.min(ptsdist, 1)
        minptsdistvalue = torch.squeeze(minptsdist[0], -1)
        nnvidx = torch.squeeze(minptsdist[1], -1)  # P
        self.smplbw = bw[nnvidx, :][None, ...]
        
        self.smpllatentdeform = nn.Embedding(cfg.num_train_frame, 128)

        D = 8
        self.deformskips = [4]
        defW = 1024
        layers = [nn.Linear(128, defW)]  # node coding + latent code
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = defW
            if i in self.deformskips:
                in_channels += 128
            layers += [layer(in_channels, defW)]

        self.smpldeformpara_linears = nn.ModuleList(layers)

        self.smpldeformpara_finallinear = nn.Linear(defW, self.smplnodenum * 6)

    def deformationsmoothloss(self):
        #smooth constrain loss
        repmodelnodepos = self.modelnodepos.unsqueeze(1).repeat(1, self.modelnodeedgenum, 1)  # nodenum*edgenum*3
        repmodelnodepos = repmodelnodepos.view([-1, 3])  # (nodenum*edgenum)*3

        self.modelnodeedge = self.modelnodeedge.view([-1])  # (nodenum*edgenum)
        relativepos = repmodelnodepos - self.modelnodepos[self.modelnodeedge, :]  # (nodenum*edgenum)*3
        relativepos = relativepos[None, ..., None]
        deformrelativepos = torch.matmul(self.deformation_affine[:, self.modelnodeedge, :, :],
                                         relativepos)  # B*(nodenum*edgenum)*3*3, #B*(nodenum*edgenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.modelnodepos[self.modelnodeedge,
                                                    :][None, ...] + self.deformation_transl[:, self.modelnodeedge, :]
        deformpos = deformpos.view(-1, self.modelnodenum*self.modelnodeedgenum, 3)  # B*(nodenum*edgenum)*3
        repnodetransl = self.deformation_transl.unsqueeze(2).repeat(1, 1, self.modelnodeedgenum, 1)  # B*nodenum*edgenum*3
        repnodetransl = repnodetransl.view([-1, self.modelnodenum*self.modelnodeedgenum, 3])  # B*(nodenum*edgenum)*3

        smoothpos = deformpos - (repmodelnodepos[None, ...] + repnodetransl)
        smoothpos = smoothpos.view(-1, self.modelnodenum, self.modelnodeedgenum, 3)# B*nodenum*edgenum*3
        smoothpos = smoothpos**2

        smoothloss = torch.sum(smoothpos)
        return smoothloss

    def deformationsmoothloss_smpl(self):
        #smooth constrain loss
        repmodelnodepos = self.smplnodepos.unsqueeze(1).repeat(1, self.smplnodeedgenum, 1)  # nodenum*edgenum*3
        repmodelnodepos = repmodelnodepos.view([-1, 3])  # (nodenum*edgenum)*3

        self.smplnodeedge = self.smplnodeedge.view([-1])  # (nodenum*edgenum)
        relativepos = repmodelnodepos - self.smplnodepos[self.smplnodeedge, :]  # (nodenum*edgenum)*3
        relativepos = relativepos[None, ..., None]
        deformrelativepos = torch.matmul(self.deformation_affine_smpl[:, self.smplnodeedge, :, :],
                                         relativepos)  # B*(nodenum*edgenum)*3*3, #B*(nodenum*edgenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.smplnodepos[self.smplnodeedge,
                                                    :][None, ...] + self.deformation_transl_smpl[:, self.smplnodeedge, :]
        deformpos = deformpos.view(-1, self.smplnodenum*self.smplnodeedgenum, 3)  # B*(nodenum*edgenum)*3
        repnodetransl = self.deformation_transl_smpl.unsqueeze(2).repeat(1, 1, self.smplnodeedgenum, 1)  # B*nodenum*edgenum*3
        repnodetransl = repnodetransl.view([-1, self.smplnodenum*self.smplnodeedgenum, 3])  # B*(nodenum*edgenum)*3

        smoothpos = deformpos - (repmodelnodepos[None, ...] + repnodetransl)
        smoothpos = smoothpos.view(-1, self.smplnodenum, self.smplnodeedgenum, 3)# B*nodenum*edgenum*3
        smoothpos = smoothpos**2

        smoothloss = torch.sum(smoothpos)
        return smoothloss
        
    def batch_rodrigues(self, rot_vecs, epsilon=1e-8, dtype=torch.float32):
        ''' Calculates the rotation matrices for a batch of rotation vectors
            Parameters
            ----------
            rot_vecs: torch.tensor Nx3
                array of N axis-angle vectors
            Returns
            -------
            R: torch.tensor Nx3x3
                The rotation matrices for the given axis-angle parameters
        '''

        batch_size = rot_vecs.shape[0]
        device = rot_vecs.device

        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle

        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)

        # Bx1 arrays
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

        zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))

        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        #rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

        t = torch.bmm(rot_dir.unsqueeze(2), rot_dir.unsqueeze(1))
        rot_mat = cos * ident + (1 - cos) * t + sin * K
        return rot_mat

    def predicting_deformation(self, sp_input):
        latent = self.latentdeform(sp_input['latent_index'].to(torch.int64))  # .type(torch.LongTensor).to(self.device)np.asscalar(np.int16(sp_input['latent_index']))
        h = latent
 
        for i, l in enumerate(self.deformpara_linears):
           h = self.deformpara_linears[i](h)
           h = F.relu(h)
           if i in self.deformskips:
               h = torch.cat([latent, h], -1)

        h = self.deformpara_finallinear(h)
        h = h.view(-1,self.modelnodenum,6)#

        # Axis–angle, Translation
        deformation_affine, deformation_transl = torch.split(h, [3, 3], dim=-1)  # 9

        deformation_rotate = self.batch_rodrigues(deformation_affine.view([-1, 3]))  # (B*nodenum)*3-->(B*nodenum)*3*3

        self.deformation_affine = deformation_rotate.view(-1, self.modelnodenum, 3, 3)#deformation_affine
        self.deformation_transl = deformation_transl.view(-1, self.modelnodenum, 3)
        return self.deformation_affine, self.deformation_transl
      
    def predicting_deformation_smpl(self, sp_input):
        latent = self.smpllatentdeform(sp_input['latent_index'].to(torch.int64))  # .type(torch.LongTensor).to(self.device)np.asscalar(np.int16(sp_input['latent_index']))
        h = latent
        for i, l in enumerate(self.smpldeformpara_linears):
           h = self.smpldeformpara_linears[i](h)
           h = F.relu(h)
           if i in self.deformskips:
               h = torch.cat([latent, h], -1)
       
        h = self.smpldeformpara_finallinear(h)
        h = h.view(-1,self.smplnodenum,6)#
       
        deformation_affine, deformation_transl = torch.split(h, [3, 3], dim=-1)  # 9
        deformation_rotate = self.batch_rodrigues(deformation_affine.view([-1, 3]))  # (B*nodenum)*3-->(B*nodenum)*3*3

        self.deformation_affine_smpl = deformation_rotate.view(-1, self.smplnodenum, 3, 3)#deformation_affine
        self.deformation_transl_smpl = deformation_transl.view(-1, self.smplnodenum, 3)
       
        return self.deformation_affine_smpl, self.deformation_transl_smpl
              
    def deformingtemplate(self):
        reptemplateshape = self.templateshape.unsqueeze(1).repeat(1, self.modelvert_nodenum, 1)  # vtnum*nodenum*3
        reptemplateshape = reptemplateshape.view([-1, 3])# (vtnum*nodenum)*3
        self.modelvert_node =self.modelvert_node.view([-1])# (vtnum*nodenum)
        relativepos = reptemplateshape - self.modelnodepos[self.modelvert_node, :]  # (vtnum*nodenum)*3
        relativepos = relativepos[None, ..., None]
        deformrelativepos = torch.matmul(self.deformation_affine[:, self.modelvert_node, :, :],
                                        relativepos)  # B*(vtnum*nodenum)*3*3, #B*(vtnum*nodenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.modelnodepos[self.modelvert_node,
                                                   :][None, ...] + self.deformation_transl[:, self.modelvert_node, :]#B*(vtnum*nodenum)*3
        deformpos = deformpos.view(-1,self.vtnum,self.modelvert_nodenum,3)#B*vtnum*nodenum*3
        weighteddeformpos = deformpos*self.modelvert_nodeweight[None, ..., None]#nodeweight: B*vtnum*nodenum*1
        weighteddeformpos = torch.sum(weighteddeformpos, dim=2)#B*vtnum*1*3
        self.deformedverts = weighteddeformpos.squeeze(2)#B*vtnum*3

        return self.deformedverts
     
    def deformingtemplate_smpl(self):
        reptemplateshape = self.templatesmpl.unsqueeze(1).repeat(1, self.smplvert_nodenum, 1)  # vtnum*nodenum*3
        reptemplateshape = reptemplateshape.view([-1, 3])# (vtnum*nodenum)*3
        self.smplvert_node =self.smplvert_node.view([-1])# (vtnum*nodenum)
        relativepos = reptemplateshape - self.smplnodepos[self.smplvert_node, :]  # (vtnum*nodenum)*3
        relativepos = relativepos[None, ..., None]
        deformrelativepos = torch.matmul(self.deformation_affine_smpl[:, self.smplvert_node, :, :],
                                        relativepos)  # B*(vtnum*nodenum)*3*3, #B*(vtnum*nodenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.smplnodepos[self.smplvert_node,
                                                   :][None, ...] + self.deformation_transl_smpl[:, self.smplvert_node, :]#B*(vtnum*nodenum)*3
        deformpos = deformpos.view(-1,self.smplvtnum,self.smplvert_nodenum,3)#B*vtnum*nodenum*3
        weighteddeformpos = deformpos*self.smplvert_nodeweight[None, ..., None]#nodeweight: B*vtnum*nodenum*1
        weighteddeformpos = torch.sum(weighteddeformpos, dim=2)#B*vtnum*1*3
        smplgraphdeformedverts = weighteddeformpos.squeeze(2)#B*vtnum*3

        return smplgraphdeformedverts    
    def deformingcloth_graphdeform_LBS(self, sp_input):

        #embedded deformation on template shape in T pose
        graphdeformedverts = self.deformingtemplate()

        #deforming with LBS of SMPL further
        sh = graphdeformedverts.shape
        
        A = torch.bmm(self.bw, sp_input['A'].view(sh[0], 24, -1))
        
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]
        pts = torch.sum(R * graphdeformedverts[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        
        self.deformedcloth = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        
        return self.deformedcloth, graphdeformedverts
        
    def deformingcloth_graphdeform_LBS_updatepose(self, sp_input):

        #embedded deformation on template shape in T pose
        graphdeformedverts = self.deformingtemplate()

        sh = graphdeformedverts.shape
        
        A = torch.bmm(self.bw, self.J_A.view(sh[0], 24, -1))
        
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]
        pts = torch.sum(R * graphdeformedverts[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        
        transl = self.displace(sp_input['latent_index'].to(torch.int64))
        h = transl.unsqueeze(1)
        self.deformedcloth = pts+ sp_input['Th']+h
        
        return self.deformedcloth, graphdeformedverts

    def deformingsmpl_graphdeform_LBS(self, sp_input):

        #embedded deformation on template shape in T pose
        self.smplgraphdeformedverts = self.deformingtemplate_smpl()

        #deforming with LBS of SMPL further
        sh = self.smplgraphdeformedverts.shape
        
        A = torch.bmm(self.smplbw, sp_input['A'].view(sh[0], 24, -1))
        
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]
        pts = torch.sum(R * self.smplgraphdeformedverts[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        
        self.deformedsmpl = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        
        _,trinormal = mesh_face_areas_normals(self.smplgraphdeformedverts.view(-1,3), self.desmplfaces)
         
        self.deformedsmpl_vertnorm = trinormal[self.desmpl_vfidx,:]
        
        return self.deformedsmpl, self.smplgraphdeformedverts
     
    def inversedeforming_samplepoints_layer(self, wpts, sp_input):
    
        #inverse deforming with smpl model, posedirs      
        smpl_ptsdist = torch.cdist(wpts, self.deformedsmpl, p=2)#deformedpersonsmpl
        smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        smpl_nnvidx = torch.squeeze(smpl_ptsdistmin[1], -1)  # B*P
        ptssmpl = self.inversedeforming_samplepoints_LBS(wpts, smpl_nnvidx, self.deformedsmpl, self.smplbw, sp_input)#posedirs, 
        
        smpl_invdeformpts = self.inversedeforming_samplepoints_graphdeform_smpl(ptssmpl, smpl_nnvidx, sp_input)
        
        #inverse deforming with cloth model 
        cloth_ptsdist = torch.cdist(wpts, self.deformedcloth, p=2)
        cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P     
        pts = self.inversedeforming_samplepoints_LBS(wpts, cloth_nnvidx, self.deformedcloth, self.bw, sp_input)
        
        cloth_invdeformpts = self.inversedeforming_samplepoints_graphdeform(pts, cloth_nnvidx, sp_input)

        return smpl_invdeformpts, cloth_invdeformpts
        
    def inversedeforming_samplepoints_LBS(self, wpts, nnvidx, templatevert, bw, sp_input):
        # inversely deforming sample points to cannonical frame
        ptsnum = wpts.size(1)
        
        templatevtnum = templatevert.size(1)

        #world points to posed points
        pts = torch.matmul(wpts - sp_input['Th'], sp_input['R'])
        #transform points from the pose space to the T pose

        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(wpts.size(0))]
        idx = torch.cat(idx).to(self.device)
        bwidx = nnvidx.view(-1) + idx
        bw1 = bw.view(-1, 24)  #self.templatebw
        selectbw = bw1[bwidx.long(), :]
        selectbw = selectbw.view(-1, ptsnum, 24)

        sh = pts.shape
        A = torch.bmm(selectbw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)#A: n_batch, 24, 4, 4
        pts = pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3])
        pts = torch.sum(R_inv * pts[:, :, None], dim=3)
        
        return pts
        
    def inversedeforming_samplepoints_graphdeform(self, pts, nnvidx, sp_input):
        #inversely deforming sample points to cannonical frame
        ptsnum = pts.size(1)

        sh = pts.shape
       
        #inverse embedded deformation
        repwpts = pts.unsqueeze(2).repeat(1, 1, self.modelvert_nodenum, 1)# B*vtnum*nodenum*3
        self.modelvert_node = self.modelvert_node.view([-1,self.modelvert_nodenum])
        ptsnode = self.modelvert_node[nnvidx.view([-1]),:] # (B*P)*nodenum
        ptsnode = ptsnode.view([-1])# (B*P*nodenum); the influence nodes for each pt

        sh = ptsnum * self.modelvert_nodenum
        idx = [torch.full([sh], i * self.modelnodenum, dtype=torch.long) for i in range(pts.size(0))]
        idx = torch.cat(idx).to(self.device)
        ptsnodeidx = ptsnode + idx# batch idx of the influence nodes, used for retrieving deformation (affine and transl) of each batch sample
        ptsnodeidx = ptsnodeidx.long()
        deformtransl = self.deformation_transl.view(-1, 3)# (B*modelnodenum)*3
        selectdeformtransl = deformtransl[ptsnodeidx, :]
        selectdeformtransl = selectdeformtransl.view(-1, ptsnum,self.modelvert_nodenum, 3)  # B*(vtnum*nodenum)*3

        repwpts = repwpts.view([-1, ptsnum,self.modelvert_nodenum, 3])
        nodepos = self.modelnodepos[ptsnode, :].view([-1, ptsnum, self.modelvert_nodenum, 3])
        relativepos = repwpts - nodepos - selectdeformtransl  # B*vtnum*nodenum*3

        deformaffine = self.deformation_affine.view(-1, 3, 3)
        selectdeformaffine = deformaffine[ptsnodeidx, :, :]  # (B*vtnum*nodenum)*3*3
        selectdeformaffine = selectdeformaffine.view(-1, ptsnum, self.modelvert_nodenum, 3, 3)  # B*vtnum*nodenum*3*3
        deformednodepos = torch.matmul(selectdeformaffine, nodepos[..., None])# B*vtnum*nodenum*3*1
        relativepos = relativepos + deformednodepos.squeeze(-1)# B*vtnum*nodenum*3

        ptsnodeweight = self.modelvert_nodeweight[nnvidx.view([-1]), :]  # (B*vtnum)*nodenum
        ptsnodeweight = ptsnodeweight.view([-1, ptsnum, self.modelvert_nodenum,1])
        weightedrelativepts = relativepos * ptsnodeweight  # B*vtnum*nodenum*3
        weightedrelativepts = torch.sum(weightedrelativepts, dim=2)  # B*vtnum*1*3
        weightedrelativepts = weightedrelativepts.squeeze(2)

        weighteddeformaffine = selectdeformaffine * ptsnodeweight[...,None]
        weighteddeformaffine = torch.sum(weighteddeformaffine, dim=2)  # B*vtnum*1*3*3
        weighteddeformaffine = weighteddeformaffine.squeeze(2)
        a = weighteddeformaffine.view(-1, 3, 3)# (B*vtnum)*3*3
        c = a.inverse()
        inversedeformaffine = c.view(-1, ptsnum, 3, 3)# B*vtnum*3*3

        deformpts = torch.matmul(inversedeformaffine, weightedrelativepts[..., None])  # B*vtnum*3*1
        deformpts = deformpts.squeeze(-1)# B*vtnum*3

        return deformpts
    
    def inversedeforming_samplepoints_graphdeform_smpl(self, pts, nnvidx, sp_input):
        #inversely deforming sample points to cannonical frame
        ptsnum = pts.size(1)

        sh = pts.shape
       
        #inverse embedded deformation
        repwpts = pts.unsqueeze(2).repeat(1, 1, self.smplvert_nodenum, 1)# B*vtnum*nodenum*3
        self.smplvert_node = self.smplvert_node.view([-1,self.smplvert_nodenum])
        ptsnode = self.smplvert_node[nnvidx.view([-1]),:] # (B*P)*nodenum
        ptsnode = ptsnode.view([-1])# (B*P*nodenum); the influence nodes for each pt

        sh = ptsnum * self.smplvert_nodenum
        idx = [torch.full([sh], i * self.smplnodenum, dtype=torch.long) for i in range(pts.size(0))]
        idx = torch.cat(idx).to(self.device)
        ptsnodeidx = ptsnode + idx# batch idx of the influence nodes, used for retrieving deformation (affine and transl) of each batch sample
        ptsnodeidx = ptsnodeidx.long()
        deformtransl = self.deformation_transl_smpl.view(-1, 3)# (B*modelnodenum)*3
        selectdeformtransl = deformtransl[ptsnodeidx, :]
        selectdeformtransl = selectdeformtransl.view(-1, ptsnum,self.smplvert_nodenum, 3)  # B*(vtnum*nodenum)*3

        repwpts = repwpts.view([-1, ptsnum,self.smplvert_nodenum, 3])
        nodepos = self.smplnodepos[ptsnode, :].view([-1, ptsnum, self.smplvert_nodenum, 3])
        relativepos = repwpts - nodepos - selectdeformtransl  # B*vtnum*nodenum*3

        deformaffine = self.deformation_affine_smpl.view(-1, 3, 3)
        selectdeformaffine = deformaffine[ptsnodeidx, :, :]  # (B*vtnum*nodenum)*3*3
        selectdeformaffine = selectdeformaffine.view(-1, ptsnum, self.smplvert_nodenum, 3, 3)  # B*vtnum*nodenum*3*3
        deformednodepos = torch.matmul(selectdeformaffine, nodepos[..., None])# B*vtnum*nodenum*3*1
        relativepos = relativepos + deformednodepos.squeeze(-1)# B*vtnum*nodenum*3

        ptsnodeweight = self.smplvert_nodeweight[nnvidx.view([-1]), :]  # (B*vtnum)*nodenum
        ptsnodeweight = ptsnodeweight.view([-1, ptsnum, self.smplvert_nodenum,1])
        weightedrelativepts = relativepos * ptsnodeweight  # B*vtnum*nodenum*3
        weightedrelativepts = torch.sum(weightedrelativepts, dim=2)  # B*vtnum*1*3
        weightedrelativepts = weightedrelativepts.squeeze(2)

        weighteddeformaffine = selectdeformaffine * ptsnodeweight[...,None]
        weighteddeformaffine = torch.sum(weighteddeformaffine, dim=2)  # B*vtnum*1*3*3
        weighteddeformaffine = weighteddeformaffine.squeeze(2)
        a = weighteddeformaffine.view(-1, 3, 3)# (B*vtnum)*3*3
        c = a.inverse()
        inversedeformaffine = c.view(-1, ptsnum, 3, 3)# B*vtnum*3*3

        deformpts = torch.matmul(inversedeformaffine, weightedrelativepts[..., None])  # B*vtnum*3*1
        deformpts = deformpts.squeeze(-1)# B*vtnum*3

        return deformpts
          
    def LBS_simulatedcloth(self, clothvert, sp_input):

        sh = clothvert.shape
        A = torch.bmm(self.bw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]#not including global rotation
        pts = torch.sum(R * clothvert[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        deformedclothvert = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']

        return deformedclothvert
          
    def update_embeddedgraph(self, clothvert):
    
        self.templateshape = clothvert
        self.vtnum = self.templateshape.size(0)
        self.modelnodepos = self.templateshape[self.modelnodeidx, :]
    
class OccupancyNetwork(nn.Module):
    def __init__(self):
        super(OccupancyNetwork, self).__init__()

        self.actvn = nn.ReLU()

        self.skips = [4]
        D = 8
        W = 256
        input_ch = 63
        input_ch_views = 27
        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += input_ch
            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)
        self.alpha_linear = nn.Linear(W, 1)
        self.alpha_linear.bias.data.fill_(0.693)

    def forward(self, light_pts):
        h = light_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([light_pts, h], -1)
        alpha = self.alpha_linear(h)
        occupancy = 1 - torch.exp(-torch.relu(alpha))
        return occupancy, h
     
class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=None):
        super(FullyConnected, self).__init__()
        net = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.2),
        ]
        for i in range(num_layers - 2):
            net.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
            ])
        net.extend([
            nn.Linear(hidden_size, output_size),
        ])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
        
class ClothSimulation(nn.Module):
    def __init__(self):
        super(ClothSimulation, self).__init__()

        templatecloth_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_vert.txt')
        templatecloth = np.loadtxt(templatecloth_path)

        vtnum = templatecloth.shape[0]
        
        self.model = FullyConnected(
            input_size=72+10+4, output_size=vtnum * 3,
            num_layers=3,
            hidden_size=1024)
            
    def forward(self, thetas, betas, gammas):
    
        pred_verts = self.model(torch.cat((thetas, betas, gammas), dim=1))
        
        return pred_verts.view(thetas.shape[0], -1, 3)
        
class ColorNetwork(nn.Module):
    def __init__(self):
        super(ColorNetwork, self).__init__()

        input_ch = 63
        input_ch_views = 27
        D = 8
        W = 256

        self.skips = [4]

        layers = [nn.Linear(input_ch, W)]#+1
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += input_ch#+1
            layers += [layer(in_channels, W)]
        self.pts_linears = nn.ModuleList(layers)

        self.views_linears = nn.ModuleList([nn.Linear(W, W // 2)])#input channel needs to change,
        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W // 2, 3)
        self.latent_fc = nn.Linear(384, 256)#384

        self.latent = nn.Embedding(cfg.num_train_frame, 128)

    def forward(self, light_pts, viewdir, sp_input):

        input_h = light_pts
        h = input_h
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_h, h], -1)
                
        features = self.feature_linear(h)

        latent = self.latent(sp_input['latent_index'])
        latent = latent[..., None].expand(*latent.shape, h.size(1))
        latent = latent.transpose(-2,-1)
        features = torch.cat((features, latent), dim=2)
        features = self.latent_fc(features)

        h = features
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)

        return rgb
               
class SDFNetwork(nn.Module):
    def __init__(self):
        super(SDFNetwork, self).__init__()

        self.sdf_network = SDFNetwork_()

    def forward(self, wpts):
        # calculate sdf
        sdf_nn_output = self.sdf_network(wpts)
        sdf = sdf_nn_output[:, 0]
        features = sdf_nn_output[:, 1:]
        return sdf, features

class SDFNetwork_(nn.Module):
    def __init__(self):
        super(SDFNetwork_, self).__init__()

        d_in = 3
        d_out = 257
        d_hidden = 256
        n_layers = 8

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        multires = 6
        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder(multires,
                                                       input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        skip_in = [4]
        bias = 0.5
        scale = 1
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)

class PointsRendererWithFrags(torch.nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor):
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images,fragments