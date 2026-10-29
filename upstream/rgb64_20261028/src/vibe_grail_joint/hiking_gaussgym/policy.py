"""32-dimensional projected recurrent RGB representation conditions ViBe adapters."""
from torch import nn
from ..hiking_perception.policy import HikingGeometryActor

class GaussGymActor(HikingGeometryActor):
    def __init__(self,*args,geometry_dim=672,**kwargs):
        super().__init__(*args,geometry_dim=geometry_dim,**kwargs)
        width=self.decoder.adapters[0].down.in_features
        self.geometry_encoder=nn.Sequential(nn.Linear(geometry_dim,32),nn.SiLU(),nn.Linear(32,width))
        nn.init.zeros_(self.geometry_encoder[-1].weight);nn.init.zeros_(self.geometry_encoder[-1].bias)


def configure_agent(agent,dimension):
    agent['actor']['class_name']='vibe_grail_joint.hiking_gaussgym.policy.GaussGymActor'
    agent['actor']['geometry_dim']=dimension
    return agent
