"""Serializable contracts shared by collection, learning and deployment."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class GeometrySpec:
    nx: int = 16
    ny: int = 12
    x_min: float = -0.4
    x_max: float = 1.2
    y_min: float = -0.5
    y_max: float = 0.5
    z_min: float = -1.4
    z_max: float = 1.0
    height_bins: int = 24
    edge_radius: float = 0.04
    visibility_tolerance: float = 0.04

    def __post_init__(self):
        if min(self.nx, self.ny) < 2 or self.height_bins < 3:
            raise ValueError('Invalid grid size')
        if not (self.x_min < self.x_max and self.y_min < self.y_max and self.z_min < self.z_max):
            raise ValueError('Invalid grid bounds')
        if self.edge_radius <= 0 or self.visibility_tolerance <= 0:
            raise ValueError('Positive geometry tolerances required')

    @property
    def cells(self):
        return self.nx * self.ny


@dataclass(frozen=True)
class ModelSpec:
    tokens: int = 21
    token_dim: int = 192
    proprio_dim: int = 64
    hidden_dim: int = 128
    temporal: bool = True
    action_dim: int = 29
    future_state_dim: int = 15  # feet xyz, shins xyz, base linear velocity, gravity-aligned yaw frame
    horizon_steps: tuple = (5, 10, 20)
    control_dt: float = 0.02

    def __post_init__(self):
        if min(self.tokens, self.token_dim, self.proprio_dim, self.hidden_dim) < 1:
            raise ValueError('Invalid model dimensions')
        if self.action_dim != 29 or self.future_state_dim != 15:
            raise ValueError('This contract is G1 29 DOF / 15 state targets')
        if tuple(sorted(set(self.horizon_steps))) != tuple(self.horizon_steps) or min(self.horizon_steps) < 1:
            raise ValueError('Horizons must be positive, unique and sorted')


def contract(geometry=None, model=None):
    return {'schema': 1, 'geometry': asdict(geometry or GeometrySpec()),
            'model': asdict(model or ModelSpec()),
            'frame': 'root-centered gravity-aligned yaw frame; heights relative to root z',
            # v1 display string retained for existing artifacts; age/elapsed is
            # the CONTROL interval (.02 s), not measured camera transport age.
            'deploy_inputs': ['Theia patch tokens', 'q_proprio', 'elapsed camera time', 'hidden state'],
            'privileged_labels_only': ['height', 'edge', 'visibility', 'future_state', 'contact']}


def parse_contract(value):
    if value['schema'] != 1:
        raise ValueError('Unknown Hiking perception contract')
    m = dict(value['model']); m['horizon_steps'] = tuple(m['horizon_steps'])
    return GeometrySpec(**value['geometry']), ModelSpec(**m)


def scene_key(tile):
    # Only physical geometry. In particular, flat tiles may retain arbitrary
    # stair-generation metadata; hashing those fields leaks identical flats.
    if 'edges' in tile:
        edges,jumps,base=tile['edges'],tile['jumps'],tile['base']
    elif tile['kind']=='flat':
        edges,jumps,base=[],[],0.
    else:
        n,h,tread,start=tile['n'],tile['height'],tile['tread'],tile['start']
        up=[start+i*tread for i in range(n)]
        if tile['kind']=='up': edges,jumps,base=up,[h]*n,0.
        elif tile['kind']=='down': edges,jumps,base=up,[-h]*n,n*h
        else:
            down=[start+n*tread+tile['plateau']+i*tread for i in range(n)]
            edges,jumps,base=up+down,[h]*n+[-h]*n,0.
    if len(edges)!=len(jumps): raise ValueError('Invalid geometry profile')
    merged={}
    for edge,jump in zip(edges,jumps):
        merged[float(edge)]=merged.get(float(edge),0.)+float(jump)
    pairs=sorted((x,j) for x,j in merged.items() if abs(j)>1e-12)
    canonical=dict(edges=[x for x,j in pairs],jumps=[j for x,j in pairs],base=float(base) if base else 0.)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def scene_split(key):
    bucket = int(key[:8], 16) % 10
    return 'test' if bucket == 0 else 'val' if bucket == 1 else 'train'


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def require_hiking(settings):
    if settings.get('foothold_experiment', {}).get('method') != 'hiking_edges':
        raise ValueError('Hiking perception refuses SSR / unspecified task configs')
