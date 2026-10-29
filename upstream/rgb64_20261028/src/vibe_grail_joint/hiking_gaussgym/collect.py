"""RGB-only synchronized collection with current physics terrain/velocity labels."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=180031)
    p.add_argument('--num-envs',type=int,default=396);p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--chunk-steps',type=int,default=128)
    a=p.parse_args()
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import torch
    from .simulation import evaluation_env,load_baseline
    from .labels import PhysicsLabels
    from ..hiking_perception.contracts import contract,GeometrySpec
    from .writer import ShardWriter
    from ..hiking_perception.control import file_hash
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    settings=json.loads(Path(a.config).read_text())
    env,settings,cases=evaluation_env(settings,a.num_envs,a.seed,duration=a.steps*.02,
                                    domain_randomization=True,collection_cases=True)
    try:
        obs,_=env.reset();actor=load_baseline(settings,obs,a.checkpoint,env.device)
        g=GeometrySpec(nx=64,ny=32,z_min=-1.,z_max=1.)
        labels=PhysicsLabels(env.unwrapped,g)
        assert tuple(env.unwrapped.scene['head_cam'].cfg.data_types)==('rgb',)
        writer=ShardWriter(a.output,dict(contract=dict(contract(g), deploy_inputs=["Theia RGB tokens", "q_proprio"], privileged_labels_only=["height", "surface_valid", "base_lin_vel_b", "base_ang_vel_b"]),seed=a.seed,settings=settings,cases=cases,
            checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=file_hash(a.checkpoint),
            scene_key_version=2,tiles=list(labels.c.cfg.random_tiles),
            scene_keys={str(int(k[:15],16)):k for k in labels.keys},
            sensor='RGB only; no depth stream',label_source='physics profile, current body-frame velocity',
            synchronization='obs[t], height[t], velocity[t], action[t], done[t]'),a.chunk_steps)
        ids=torch.arange(a.num_envs,device=env.device)
        alive=torch.ones(a.num_envs,dtype=torch.bool,device=env.device);reset=alive.clone()
        for step in range(a.steps):
            with torch.no_grad():
                target=labels.sample(ids);c=labels.c;tile=c.tile[ids]
                root_x=c.robot.data.root_link_pos_w[ids,0]-c.tile_origins[tile,0]
                active=torch.isfinite(labels.edges[tile])&(labels.jumps[tile].abs()>1e-6)
                first=torch.where(active,labels.edges[tile],float('inf')).amin(-1)
                last=torch.where(active,labels.edges[tile],-float('inf')).amax(-1)
                stage=torch.where(root_x<first-.4,0,torch.where(root_x<first,1,torch.where(root_x<=last,2,3)))
                stage=torch.where(active.any(-1),stage,4)
                row={k:target[k].clone() for k in ('height','surface_valid','base_lin_vel_b','base_ang_vel_b','scene_id')}
                row.update(tokens=obs['kv_tokens']['img_tokens'].clone().half(),proprio=obs['q_proprio'].clone(),
                           reset=reset.clone(),valid=alive.clone(),terrain_stage=stage,target_speed=c.speed[ids].clone())
                action=actor(obs,stochastic_output=False)
                if not torch.isfinite(action).all():raise RuntimeError('Nonfinite collection action')
                obs,_,done,_=env.step(action)
                fall=env.unwrapped.scene['robot'].data.projected_gravity_b[:,2]>-.4
                writer.add(row)
                alive &= ~(done.bool()|fall);reset=done.bool()
            if step%100==0:print(json.dumps(dict(step=step,alive=int(alive.sum()))),flush=True)
        writer.close(rgb_only=True,velocity_targets='current body frame, uncorrupted physics',total_steps=a.steps)
    finally:env.close()

if __name__=='__main__':main()
