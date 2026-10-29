"""GaussGym-style recurrent RGB representation; privileged values are targets only."""
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from .decoder import ConvDecoder3D
from .voxel import heightmap_to_voxels, voxels_to_heightmap
from ..hiking_perception.contracts import GeometrySpec

ARCHITECTURE = 'hiking_gaussgym_rgb_v1'


class GaussGeometry(nn.Module):
    """Frozen Theia tokens -> 128 image features + 32 proprio -> LSTM512 + skip.

    The proprio subset matches GaussGym's gravity/joint-position choice for G1:
    3 gravity + 29 joint positions. Gyro/joint velocities are NOT encoder inputs.
    Other original ViBe actor observations remain unchanged.
    """
    hidden_dim = 512
    input_dim = 160
    latent_dim = 672

    def __init__(self, geometry=None):
        super().__init__()
        self.geometry = geometry or GeometrySpec(nx=64, ny=32, z_min=-1., z_max=1.)
        g = self.geometry
        self.visual = nn.Sequential(nn.LayerNorm(192), nn.Flatten(1), nn.Linear(21*192,128))
        self.register_buffer('proprio_mean', torch.zeros(32))
        self.register_buffer('proprio_var', torch.ones(32))
        self.register_buffer('proprio_count', torch.tensor(0.))
        self.memory = nn.LSTMCell(self.input_dim,self.hidden_dim)
        self.post = nn.Sequential(nn.Linear(self.latent_dim,512),nn.SiLU(),
                                  nn.Linear(512,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.decoder = ConvDecoder3D(256,(g.nx,g.ny,g.height_bins),256,mults=(2,3),act_final=True)
        self.occupancy = nn.Conv3d(256,1,1)
        self.centroid = nn.Sequential(nn.Conv3d(256,1,1),nn.Sigmoid())
        self.linvel = nn.Linear(256,3)
        self.angvel = nn.Linear(256,3)

    @staticmethod
    def select_proprio(proprio):
        return torch.cat((proprio[..., :3],proprio[..., 6:35]),-1)

    @torch.no_grad()
    def update_normalizer(self, proprio):
        x = self.select_proprio(proprio).flatten(0,-2)
        count = x.shape[0]; total = self.proprio_count+count
        mean = x.mean(0); var = x.var(0,unbiased=False); delta = mean-self.proprio_mean
        new_var=(self.proprio_count*self.proprio_var+count*var+delta.square()*self.proprio_count*count/total)/total
        self.proprio_mean.add_(delta*count/total)
        self.proprio_var.copy_(new_var); self.proprio_count.copy_(total)

    def initial(self,batch,device):
        return (torch.zeros(batch,512,device=device),torch.zeros(batch,512,device=device))

    def step(self,tokens,proprio,h,c,reset):
        # No geometry/depth/target argument can enter this function.
        p=(self.select_proprio(proprio)-self.proprio_mean)/self.proprio_var.clamp_min(1e-4).sqrt()
        x=torch.cat((self.visual(tokens),p),-1)
        x=x.sign()*torch.log1p(x.abs())
        keep=(~reset.bool()).to(h.dtype).unsqueeze(-1)
        h,c=self.memory(x,(h*keep,c*keep))
        return torch.cat((h,x),-1),h,c

    def sequence(self,tokens,proprio,reset,initial=None):
        h,c=self.initial(tokens.shape[1],tokens.device) if initial is None else initial
        values=[]
        for t in range(len(tokens)):
            z,h,c=self.step(tokens[t],proprio[t],h,c,reset[t]); values.append(z)
        return torch.stack(values),(h,c)

    def heads(self,latent):
        x=self.post(latent)
        grid=self.decoder(x)
        return self.occupancy(grid).squeeze(1),self.centroid(grid).squeeze(1),self.linvel(x),self.angvel(x)

    def contract(self):
        return dict(architecture=ARCHITECTURE,geometry=asdict(self.geometry),hidden_dim=512,latent_dim=672,
                    visual_input='frozen Theia RGB tokens [21,192]',proprio_input='gravity[3], relative joint positions[29]',
                    privileged_inputs=False,depth_input=False,skip='concatenate processed observations',
                    targets=['full unsaturated terrain','base linear velocity in body frame','base angular velocity in body frame'])


def load_model(state,device):
    if state['architecture']!=ARCHITECTURE: raise ValueError('Wrong perception architecture')
    model=GaussGeometry(GeometrySpec(**state['contract']['geometry'])).to(device)
    if state['contract']!=model.contract(): raise ValueError('Perception input contract mismatch')
    model.load_state_dict(state['encoder'],strict=True)
    return model


def mask_mean(value,mask):
    return torch.where(mask,value,0.).sum()/mask.sum().clamp_min(1)


def reconstruction_loss(model, latent, target):
    g=model.geometry
    logits,centroid,linvel,angvel=model.heads(latent)
    height=target['height'].reshape(-1,g.nx,g.ny)
    finite=torch.isfinite(height)
    occupancy,centroid_target=heightmap_to_voxels(torch.nan_to_num(height),g.height_bins,g.z_min,g.z_max)
    valid=target['surface_valid'].reshape_as(height).bool() & finite & (height>g.z_min) & (height<g.z_max)
    valid &= target['valid'].bool().reshape(-1,1,1)
    occupancy_loss=mask_mean(F.binary_cross_entropy_with_logits(logits.float(),occupancy.float(),reduction='none'),
                              valid[...,None].expand_as(logits))
    offset_loss=mask_mean((centroid.float()-centroid_target).square(),valid[...,None]&occupancy)
    lv=mask_mean((linvel.float()-target['base_lin_vel_b']).square().mean(-1),target['valid'].bool())
    av=mask_mean((angvel.float()-target['base_ang_vel_b']).square().mean(-1),target['valid'].bool())
    loss=occupancy_loss+offset_loss+lv+av
    with torch.no_grad():
        predicted=decode_height(logits,centroid,g)
        mae=mask_mean((predicted-height).abs(),valid)
    return loss,dict(occupancy_loss=float(occupancy_loss.detach()),offset_loss=float(offset_loss.detach()),
                     linvel_mse=float(lv.detach()),angvel_mse=float(av.detach()),height_mae_m=float(mae),
                     valid_cells=int(valid.sum()))


def decode_height(logits,centroid,g):
    occupancy=F.one_hot(logits.argmax(-1),g.height_bins).bool()
    return voxels_to_heightmap(occupancy,torch.where(occupancy,centroid,0.),g.z_min,g.z_max)
