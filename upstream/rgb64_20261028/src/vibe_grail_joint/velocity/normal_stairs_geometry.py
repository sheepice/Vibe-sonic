"""Normal-depth stair profiles and Hiking-style sharp triangular-mesh edges.

The triangle surface represents the exterior of the SAME step profile used by
physics. Internal box seams and coplanar triangle diagonals are not obstacles.
"""
import itertools
import math
import numpy as np
import torch


def normal_tiles(seed, levels=11, variants=3, minimum_tread=.28, maximum_tread=.40):
    if not .25 <= minimum_tread <= maximum_tread:
        raise ValueError('Normal stair treads must fit the G1 sole with a margin')
    rng = np.random.default_rng(seed)
    tiles = []
    for level, variant, kind in itertools.product(range(levels), range(variants), ('up', 'down', 'up_down', 'flat')):
        n = int(rng.integers(4, 9))
        maximum = .04 + .02 * level
        if variant == 0:
            risers = np.full(n, rng.uniform(.95, 1.05) * maximum)
            treads = np.full(n, rng.uniform(minimum_tread, maximum_tread))
        else:
            risers = rng.uniform(.70, 1.05, n) * maximum
            treads = rng.uniform(minimum_tread, maximum_tread, n)
        risers = np.clip(risers, .025, .25)
        start, plateau = float(rng.uniform(1.2, 1.8)), float(rng.uniform(.9, 1.4))
        up = start + np.r_[0., np.cumsum(treads[:-1])]
        end_up = start + treads.sum()
        down = end_up + plateau + np.r_[0., np.cumsum(treads[::-1][:-1])]
        if kind == 'up':
            edges, jumps, base, end = up, risers, 0., end_up
        elif kind == 'down':
            edges, jumps, base, end = up, -risers[::-1], float(risers.sum()), end_up
        elif kind == 'up_down':
            edges, jumps, base, end = np.r_[up, down], np.r_[risers, -risers[::-1]], 0., end_up + plateau + treads.sum()
        else:
            edges, jumps, base, end = np.array([]), np.array([]), 0., 6.
        i = len(tiles)
        tiles.append(dict(level=level, variant=variant, kind=kind, n=n,
                          height=float(risers.max()), tread=float(treads.mean()),
                          start=start, plateau=plateau, goal=float(end + .6),
                          origin=[-30. - (i % 12)*16., -30. - (i // 12)*10., 0.],
                          edges=edges.tolist(), jumps=jumps.tolist(), base=base,
                          risers=risers.tolist(), treads=treads.tolist(), total_height=float(risers.sum())))
    return tiles


def profile_mesh(tile, half_width=2.):
    """Watertight triangulated extrusion, with welded vertices and no box seams."""
    x = np.array([-2., *tile['edges'], 12.], dtype=float)
    h = np.maximum(np.r_[tile['base'], tile['base']+np.cumsum(tile['jumps'])], 0.)
    h = np.round(h, 10)
    z_levels = np.unique(np.r_[-.1, 0., h])
    vertices, faces, lookup = [], [], {}

    def quad(points, normal):
        p = np.asarray(points, dtype=float)
        if np.dot(np.cross(p[1]-p[0], p[2]-p[0]), normal) < 0:
            p = p[::-1]
        ids = []
        for value in p:
            key = tuple(np.round(value, 10))
            if key not in lookup:
                lookup[key] = len(vertices)
                vertices.append(value)
            ids.append(lookup[key])
        faces.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))

    for i, top in enumerate(h):
        a, b = x[i:i+2]
        quad([(a,-half_width,top),(b,-half_width,top),(b,half_width,top),(a,half_width,top)], [0,0,1])
        quad([(a,-half_width,-.1),(b,-half_width,-.1),(b,half_width,-.1),(a,half_width,-.1)], [0,0,-1])
        levels = z_levels[z_levels <= top+1e-9]
        for lo, hi in zip(levels[:-1], levels[1:]):
            for y in (-half_width, half_width):
                quad([(a,y,lo),(b,y,lo),(b,y,hi),(a,y,hi)], [0,np.sign(y),0])
    for i, at in enumerate(x):
        left = h[i-1] if i else -.1
        right = h[i] if i < len(h) else -.1
        lo, hi = sorted((left,right))
        levels = z_levels[(z_levels >= lo-1e-9) & (z_levels <= hi+1e-9)]
        for bottom, top in zip(levels[:-1], levels[1:]):
            quad([(at,-half_width,bottom),(at,half_width,bottom),(at,half_width,top),(at,-half_width,top)],
                 [-1 if right > left else 1,0,0])
    return np.array(vertices, dtype=float), np.array(faces, dtype=np.int64)


def sharp_mesh_edges(vertices, faces, threshold_deg=45., minimum_z=.001):
    """Paper Algorithm 1: adjacent face angle, then concatenate collinear edges."""
    tri = vertices[faces]
    normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('Degenerate triangle')
    normals /= lengths[:,None]
    adjacency = {}
    for fid, face in enumerate(faces):
        for a, b in zip(face, np.roll(face,-1)):
            adjacency.setdefault(tuple(sorted((int(a),int(b)))), []).append(fid)
    raw = []
    for (a,b), adjacent in adjacency.items():
        if len(adjacent) != 2:
            raise ValueError('Edge extraction requires a watertight manifold triangle surface')
        if normals[adjacent[0]] @ normals[adjacent[1]] < math.cos(math.radians(threshold_deg)):
            segment = vertices[[a,b]]
            if segment[:,2].max() > minimum_z:
                raw.append(segment)
    return concatenate_edges(np.array(raw).reshape(-1,2,3))


def concatenate_edges(segments, tolerance=1e-6):
    groups = {}
    for a,b in segments:
        direction = (b-a) / np.linalg.norm(b-a)
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
        origin = a-direction*np.dot(a,direction)
        key = tuple(np.round(np.r_[direction,origin],6))
        groups.setdefault(key, (direction,origin,[]))[2].append(sorted((float(a@direction),float(b@direction))))
    result = []
    for direction, origin, intervals in groups.values():
        current = None
        for lo,hi in sorted(intervals):
            if current is None:
                current = [lo,hi]
            elif lo <= current[1]+tolerance:
                current[1] = max(current[1],hi)
            else:
                result.append([origin+current[0]*direction,origin+current[1]*direction])
                current = [lo,hi]
        result.append([origin+current[0]*direction,origin+current[1]*direction])
    return np.array(result,dtype=float).reshape(-1,2,3)


def cylinder_penetration(points, segments, valid, radius):
    """Exact capped-cylinder SDF; points [B,...,3], segments [B,E,2,3]."""
    original = points.shape[:-1]
    p = points.flatten(1,-2)
    a, b = segments[:,:,0], segments[:,:,1]
    delta = b-a
    length = delta.norm(dim=-1).clamp_min(1e-8)
    axis = delta/length[...,None]
    relative = p[:,:,None]-((a+b)/2)[:,None]
    axial = (relative*axis[:,None]).sum(-1)
    radial = (relative-axial[...,None]*axis[:,None]).norm(dim=-1)-radius
    cap = axial.abs()-length[:,None]/2
    q = torch.stack((radial,cap),-1)
    sdf = q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
    depth = (-sdf).clamp_min(0)*valid[:,None]
    return depth.amax(-1).reshape(original)


def foot_volume_points(device='cpu'):
    """Dense volume grid filtered by the G1 foot's seven 1 cm capsules.

    Maximum longitudinal spacing is 18 mm; sparse endpoint-only sampling would
    leave holes in the detector even while an edge crosses the middle of a foot.
    """
    ends = torch.tensor([
        [[.1,-.026,-.025],[.05,-.027,-.025]],
        [[-.044,-.018,-.025],[.123,-.018,-.025]],
        [[-.052,-.010,-.025],[.130,-.010,-.025]],
        [[-.054,0.,-.025],[.132,0.,-.025]],
        [[-.052,.010,-.025],[.130,.010,-.025]],
        [[-.044,.018,-.025],[.123,.018,-.025]],
        [[.1,.026,-.025],[.05,.026,-.025]],
    ],device=device)
    grid = torch.cartesian_prod(torch.linspace(-.06,.138,12,device=device),
                                torch.linspace(-.03,.03,4,device=device),
                                torch.tensor([-.034,-.025,-.016],device=device))
    delta=ends[:,1]-ends[:,0]
    relative=grid[:,None]-ends[None,:,0]
    fraction=((relative*delta).sum(-1)/delta.square().sum(-1)).clamp(0,1)
    distance=(relative-fraction[...,None]*delta).norm(dim=-1)
    return grid[(distance<=.0100001).any(-1)]
