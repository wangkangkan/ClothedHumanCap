import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer_occupancy
import os
import numpy as np
import scipy.sparse
import scipy.io as scio
from lib.config import cfg
from chumpy.utils import row, col

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = if_clight_renderer_occupancy.Renderer(self.net)

        # Fix the clothing shape and the geometry network
        for param in self.net.cloth_simulation.parameters():
             param.requires_grad = False
       
        for param in self.net.occupancy_network_smpl.parameters():
             param.requires_grad = False
        
        self.net.tempclothpara.weight.requires_grad = False  
        for param in self.net.occupancy_network_cloth.parameters():
             param.requires_grad = False             
        for param in self.net.sdf_network_cloth.parameters():
             param.requires_grad = False
        
        for param in self.net.deformation_network1.parameters():
            param.requires_grad = False
        for param in self.net.deformation_network2.parameters():
            param.requires_grad = False
            
        self.acc_crit = torch.nn.functional.smooth_l1_loss

        self.msk2mse = lambda x, y: torch.mean((x - y) ** 2)
        
        self.sdf_crit = torch.nn.L1Loss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.updateswtag = 1
    
    def img2mse(self, x, y, M=None):
        if M == None:
            return torch.mean((x - y) ** 2)
        else:
            return torch.sum((x - y) ** 2 * M) / (torch.sum(M) + 1e-8) / x.shape[-1]
     
    def forward(self, batch, epoch):
        ret = self.renderer.render_deformation(batch)
        
        scalar_stats = {}
        loss = 0

        img_loss = self.img2mse(ret['rgb_map'], batch['rgb'])
        scalar_stats.update({'img_loss': img_loss})
        loss += 1.0 *img_loss

        scalar_stats.update({'IoUloss': self.renderer.IoUloss})
        loss += 10.0 *self.renderer.IoUloss #10

        scalar_stats.update({'IoUloss_def': self.renderer.IoUloss_def})
        loss += 30.0 *self.renderer.IoUloss_def #30
        
        #--------------------
        scalar_stats.update({'graphdeform_loss': self.renderer.graphdeform_loss})
        loss += 0.01 *self.renderer.graphdeform_loss #0.001
        
        scalar_stats.update({'attach_loss': self.renderer.attach_loss})
        loss += 0.1 *self.renderer.attach_loss #0.1
        
        smplimg_loss = self.img2mse(ret['rgb_map_s'], batch['rgb'], 1-self.renderer.coordclothrendermask)#coord_silhouette
        scalar_stats.update({'smplimg_loss': smplimg_loss})
        loss += 1.0 *smplimg_loss
        
        clothimg_loss = self.img2mse(ret['rgb_map_d'], batch['rgb'], self.renderer.coord_silhouette)
        scalar_stats.update({'clothimg_loss': clothimg_loss})
        loss += 1.0 *clothimg_loss

        #---------------
        loss += 0.05 * self.renderer.smoothloss#5.0
        scalar_stats.update({'smoothloss': self.renderer.smoothloss})
        
        loss += 0.1 * self.renderer.smoothloss_smpl#5.0
        scalar_stats.update({'smoothloss_smpl': self.renderer.smoothloss_smpl})
        
        loss += 10.0 * self.renderer.interpenetrationloss#1.0
        scalar_stats.update({'interpenetrationloss': self.renderer.interpenetrationloss})
        
        loss += 100.0 * self.renderer.interploss_graphdeform#30.0
        scalar_stats.update({'interploss_graphdeform': self.renderer.interploss_graphdeform})

        scalar_stats.update({'loss': loss})
        
        #---------------
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
