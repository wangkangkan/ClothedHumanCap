from .nerf_net_utils import *
from . import if_clight_renderer_mmsk_occupancy

class Renderer(if_clight_renderer_mmsk_occupancy.Renderer):
    def __init__(self, net):
        super(Renderer, self).__init__(net)
