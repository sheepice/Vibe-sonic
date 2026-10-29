"""Fixed 48-case evaluation, visual interventions and world-fixed RGB videos."""
import argparse
from collections import deque
import json
from pathlib import Path


class VisualIntervention:
    def __init__(self, mode='normal', delay=5):
        self.mode, self.delay = mode, delay
        self.history=deque(maxlen=delay+1)

    def __call__(self, obs):
        obs=obs.clone()
        pair=(obs['kv_tokens']['img_tokens'].clone(),obs['q_cls'].clone())
        if self.mode!='freeze' or not self.history: self.history.append(pair)
        if self.mode in ('freeze','delay'):
            obs['kv_tokens']['img_tokens'],obs['q_cls']=self.history[0]
        elif self.mode=='zero_tokens':
            obs['kv_tokens']['img_tokens'].zero_(); obs['q_cls'].zero_()
        elif self.mode=='swap_worlds':
            obs['kv_tokens']['img_tokens']=pair[0].roll(12,0); obs['q_cls']=pair[1].roll(12,0)
        return obs


def load_control(settings,obs,checkpoint,raw):
    import torch
    from .runtime import RGBMemory
    from .policy import configure_agent,GaussGymActor
    from ..velocity.config import agent_config
    state=torch.load(checkpoint,map_location=raw.device,weights_only=False)
    if 'gaussgym' not in state.get('infos',{}):
        from .simulation import load_baseline
        return load_baseline(settings,obs,checkpoint,raw.device),None
    memory=RGBMemory(state['infos']['gaussgym'],raw.num_envs,raw.device)
    cfg=configure_agent(agent_config(settings['bank_root']+'/rank_0',settings,20),memory.model.latent_dim)
    obs=obs.clone(False);obs['hiking_latent']=torch.zeros(raw.num_envs,memory.model.latent_dim,device=raw.device)
    acfg=dict(cfg['actor']);acfg.pop('class_name')
    actor=GaussGymActor(obs=obs,obs_groups=cfg['obs_groups'],obs_set='actor',output_dim=29,**acfg).to(raw.device).eval()
    actor.load_state_dict(state['actor_state_dict'],strict=True)
    return actor,memory


def augment(obs,memory,raw,reset):
    if memory is None:return obs
    obs=obs.clone(False);obs['hiking_latent']=memory.step(obs,reset)
    return obs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--duration',type=float,default=40.); p.add_argument('--seed',type=int,default=170017)
    p.add_argument('--intervention',choices=('normal','zero_tokens','freeze','delay','swap_worlds','zero_geometry','swap_geometry'),default='normal')
    p.add_argument('--delay-steps',type=int,default=5); p.add_argument('--record-case',type=int)
    p.add_argument('--irregular',action='store_true'); p.add_argument('--num-envs',type=int,default=48)
    p.add_argument('--approach-extra',type=float,default=0.,help='Shift stairs forward by 0..2 m while preserving initial robot/reference')
    args=p.parse_args()
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import torch
    import numpy as np
    from .simulation import evaluation_env
    from ..hiking_perception.contracts import write_json
    from ..hiking_perception.control import file_hash
    from ..velocity.posture_diagnostics import PostureDiagnostics
    from ..velocity.posture_summary import PostureSummary
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    env,settings,cases=evaluation_env(json.loads(Path(args.config).read_text()),args.num_envs,args.seed,
        uniform=not args.irregular,duration=args.duration,record_case=args.record_case,approach_extra=args.approach_extra)
    video=None
    try:
        obs,_=env.reset(); raw=env.unwrapped; n=raw.num_envs; robot=raw.scene['robot']; c=raw.command_manager.get_term('motion')
        actor,memory=load_control(settings,obs,args.checkpoint,raw)
        if args.intervention in ('zero_geometry','swap_geometry') and memory is None:
            raise ValueError('Geometry ablation requires a geometry-conditioned actor')
        intervention=VisualIntervention(args.intervention,args.delay_steps)
        reset=torch.ones(n,dtype=torch.bool,device=env.device)
        initial_state=dict(root=robot.data.root_link_pos_w.cpu().tolist(),joints=robot.data.joint_pos.cpu().tolist())
        active=np.ones(n,bool); held=np.zeros(n,int); rows=[None]*n; trace=[]
        speed_error=np.zeros(n); steps=np.zeros(n,int)
        tiles=[c.cfg.random_tiles[case['tile']] for case in cases]; goals=np.array([t['goal'] for t in tiles])
        posture=PostureDiagnostics(c); summary=PostureSummary(n,env.device,keep_trace=False)
        camera_pose=None; max_camera_drift=0.
        if args.record_case is not None:
            import imageio.v2 as iio
            from .labels import PhysicsLabels as GeometryLabels
            from .runtime import RGBMemory as PerceptionState
            from .model import decode_height
            preview_labels=GeometryLabels(raw,memory.model.geometry) if isinstance(memory,PerceptionState) else GeometryLabels(raw)
            video=iio.get_writer(out/'fixed_rgb.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1)
            ci=raw.scene['review_camera'].camera_idx
            camera_pose=(raw.sim.data.cam_xpos[:,ci].clone(),raw.sim.data.cam_xmat[:,ci].clone())
        with torch.inference_mode():
            for step in range(round(args.duration/raw.step_dt)):
                acted=augment(intervention(obs),memory,raw,reset)
                if args.intervention=='zero_geometry': acted['hiking_latent'].zero_()
                if args.intervention=='swap_geometry': acted['hiking_latent']=acted['hiking_latent'].roll(12,0)
                if video is not None and step%2==0:
                    # Freeze all panels at obs[t], before physics advances to t+1.
                    frame=raw.scene['review_camera'].data.rgb[0].cpu().numpy().copy()
                    rgb=raw.scene['head_cam'].data.rgb[0].cpu().numpy().copy()
                    target=preview_labels.sample(torch.zeros(1,dtype=torch.long,device=env.device))
                    prediction=None
                    if isinstance(memory,PerceptionState):
                        logits,offset,_,_=memory.model.heads(memory.latent[:1])
                        prediction=decode_height(logits,offset,memory.model.geometry)[0].flatten().cpu().numpy()
                action=actor(acted,stochastic_output=False)
                if not torch.isfinite(action).all(): raise RuntimeError('Nonfinite action')
                active_t=torch.as_tensor(active,device=env.device)
                summary.update(posture.sample(elapsed_dt_s=raw.step_dt),step*raw.step_dt,active_t)
                obs,_,done,_=env.step(action); reset=done.bool()
                if done.any(): raise RuntimeError('Unexpected auto-reset in fixed-duration evaluation')
                q=raw.quality; geom=q.geometry()
                record={k:v.cpu().numpy().copy() for k,v in {**q.step_data,**geom}.items()}
                record.update(action=action.cpu().numpy(),active=active.copy())
                trace.append(record)
                root=robot.data.root_link_pos_w; local=(root-c.tile_origins[c.tile]).cpu().numpy()
                feet=c.robot_body_pos_w[:,q.feet]; ground=c.ground_height(feet[:,:,:2])
                gravity=robot.data.projected_gravity_b[:,2].cpu().numpy()
                root_height=(root[:,2]-c.ground_height(root[:,:2])).cpu().numpy()
                passed=(local[:,0]>=goals)&((feet[:,:,0]-c.tile_origins[c.tile,None,0]).amin(1).cpu().numpy()>=goals-.45)
                stable=((feet[:,:,2]-ground).abs().amax(1).cpu().numpy()<.18)&(gravity<-.85)&(abs(local[:,1])<1.5)&(root_height>.45)
                held=np.where(passed&stable,held+1,0)
                fallen=(gravity>-.4)|(root_height<.25)
                velocity=robot.data.root_link_lin_vel_w[:,:2].cpu().numpy()
                speed_error+=np.linalg.norm(velocity-np.array([[x['speed'],0] for x in cases]),axis=1)*active; steps+=active
                terminal=active&((held>=25)|fallen|(step==round(args.duration/raw.step_dt)-1))
                for i in np.flatnonzero(terminal):
                    rows[i]=dict(case=cases[i],success=bool(held[i]>=25),fall=bool(fallen[i]),steps=step+1,time_s=(step+1)*raw.step_dt,
                                 progress_m=float(local[i,0]),speed_error_m_s=float(speed_error[i]/steps[i]))
                active[terminal]=False
                if video is not None and step%2==0:
                    from PIL import Image,ImageDraw
                    image=Image.new('RGB',(640,552),(18,23,32))
                    image.paste(Image.fromarray(frame),(0,30))
                    image.paste(Image.fromarray(rgb).resize((224,126)),(8,412))
                    draw=ImageDraw.Draw(image); draw.text((8,8),f'Hiking {args.intervention} | {cases[0]["speed"]} m/s | t={step*.02:.2f}s',fill='white')
                    draw.text((8,394),'Physical head RGB',fill='white')
                    spec=preview_labels.spec
                    def panel(values):
                        v=np.clip((values-spec.z_min)/(spec.z_max-spec.z_min),0,1)
                        color=np.stack((v,1-abs(2*v-1),1-v),-1)*255
                        color[~target['surface_valid'][0].cpu().numpy()]=25
                        return Image.fromarray(color.astype(np.uint8).reshape(spec.nx,spec.ny,3).transpose(1,0,2)[::-1].copy()).resize((192,126))
                    image.paste(panel(target['height'][0].cpu().numpy()),(240,412))
                    draw.text((240,394),'GT height | physics',fill='white')
                    if prediction is not None:
                        image.paste(panel(prediction),(440,412))
                        draw.text((440,394),'Predicted height',fill='white')
                    draw.text((240,540),'z relative to root: -1.0m blue / +1.0m red',fill='white')
                    video.append_data(np.asarray(image))
                    max_camera_drift=max(max_camera_drift,float((raw.sim.data.cam_xpos[:,ci]-camera_pose[0]).abs().max()),float((raw.sim.data.cam_xmat[:,ci]-camera_pose[1]).abs().max()))
                if step%200==0: print(json.dumps(dict(step=step,active=int(active.sum()))),flush=True)
                if not active.any(): break
        arrays={k:np.stack([r[k] for r in trace]) for k in trace[0]}
        np.savez_compressed(out/'trace.npz',**arrays)
        for i,row in enumerate(rows):
            a={k:v[:row['steps'],i] for k,v in arrays.items()}; support=a['stable_support']; margins=a['sole_edge_margin_m'][support]
            foot=a['foot_riser_force_n']>15; shin=a['shin_force_n']>15
            first=arrays['root_local'][:row['steps'],i,0] < tiles[i]['edges'][0]+.45 if tiles[i]['edges'] else np.zeros(row['steps'],bool)
            row.update(foot_impulse_ns=float(a['foot_riser_impulse_ns'].sum()),shin_impulse_ns=float(a['shin_impulse_ns'].sum()),
                first_step_collision=bool((foot[first]|shin[first]).any()),collision_free=not bool((foot|shin).any()),
                slip_distance_m=float(a['slip_distance_m'].sum()),support_slip_speed_m_s=float(a['slip_distance_m'].sum()/max(a['support_seconds'].sum(),1e-8)),
                edge_margin_p05_m=float(np.quantile(margins,.05)) if len(margins) else None,
                support_overhang_fraction=float((margins<0).mean()) if len(margins) else None)
        write_json(out/'report.json',dict(args=vars(args),checkpoint_sha256=file_hash(args.checkpoint),cases=rows,initial_state=initial_state,tiles=tiles,completed=sum(r['success'] for r in rows),total=n,
            collision_free_completed=sum(r['success'] and r['collision_free'] for r in rows),camera_max_pose_drift=max_camera_drift,
            note='All scheduled cases included; feature interventions are not equivalent to black RGB pixels. First-step window uses root x, not exact contact-to-edge attribution.'))
        write_json(out/'posture.json',summary.finalize())
        print(json.dumps(dict(output=str(out),completed=sum(r['success'] for r in rows),total=n)),flush=True)
    finally:
        if video is not None: video.close()
        env.close()


if __name__=='__main__': main()
