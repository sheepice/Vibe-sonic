"""Four independent worlds: two speeds, matched normal and pixel-black RGB."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for k in ('config','checkpoint','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--record-case',type=int,required=True)
    p.add_argument('--duration',type=float,default=40.)
    p.add_argument('--num-envs',type=int,default=4)
    args=p.parse_args()
    from .review_shared import verify_review_worker
    verify_review_worker()
    import torch,numpy as np,imageio.v2 as iio
    from PIL import Image,ImageDraw
    from .suite_scene import evaluation_env
    from .review_evaluate import load_control,augment
    from .black_rgb import BlackRGB
    from ..hiking_perception.control import file_hash
    torch.set_num_threads(2);torch.manual_seed(170017)
    settings=json.loads(Path(args.config).read_text())
    scene=json.loads(Path(settings['evaluation_suite']).read_text())[args.record_case]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    env,settings,cases=evaluation_env(settings,scene,duration=args.duration)
    writers=[]
    try:
        obs,_=env.reset();raw=env.unwrapped;c=raw.command_manager.get_term('motion');robot=raw.scene['robot']
        actor,memory=load_control(settings,obs,args.checkpoint,raw)
        black=BlackRGB(raw);masked=torch.tensor([False,True,False,True],device=raw.device)
        initial=dict(root=robot.data.root_link_pos_w.cpu().tolist(),joints=robot.data.joint_pos.cpu().tolist())
        for a,b in ((0,1),(2,3)):
            for key in initial:assert np.allclose(initial[key][a],initial[key][b],rtol=0,atol=1e-6)
        for case in cases:
            d=out/f'{case["speed"]:.1f}_{case["intervention"]}';d.mkdir()
            writers.append(iio.get_writer(d/'fixed_rgb.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1))
        reset=torch.ones(4,dtype=torch.bool,device=raw.device)
        active=np.ones(4,bool);held=np.zeros(4,int);rows=[None]*4;trace=[]
        error=np.zeros(4);counts=np.zeros(4,int)
        ci=raw.scene['review_camera'].camera_idx
        camera_pose=(raw.sim.data.cam_xpos[:,ci].clone(),raw.sim.data.cam_xmat[:,ci].clone());drift=0.
        if scene.get('heightfield'):
            from .suite_heightfield import audit_field
            (out/'heightfield_audit.json').write_text(json.dumps(audit_field(raw),indent=2))
        with torch.inference_mode():
            for step in range(round(args.duration/raw.step_dt)):
                acted=obs.clone()
                acted['kv_tokens']['img_tokens'][masked]=black.tokens[masked]
                acted['q_cls'][masked]=black.cls[masked]
                acted=augment(acted,memory,raw,reset)
                if step%2==0:
                    frames=raw.scene['review_camera'].data.rgb.cpu().numpy().copy()
                    rgbs=raw.scene['head_cam'].data.rgb.cpu().numpy().copy();rgbs[[1,3]]=0
                    # Diagnostic predictions only; ground truth is not fed into policy.
                    from .model import decode_height
                    logits,offset,_,_=memory.model.heads(memory.latent)
                    predicted=decode_height(logits,offset,memory.model.geometry).cpu().numpy()
                action=actor(acted,stochastic_output=False)
                if not torch.isfinite(action).all():raise RuntimeError('Nonfinite action')
                before=active.copy()
                obs,_,done,_=env.step(action);reset=done.bool()
                if done.any():raise RuntimeError('Unexpected reset')
                q=raw.quality;geom=q.geometry()
                record={k:v.cpu().numpy().copy() for k,v in {**q.step_data,**geom}.items()}
                record.update(active=before,action=action.cpu().numpy());trace.append(record)
                root=robot.data.root_link_pos_w;local=(root-c.tile_origins[c.tile]).cpu().numpy()
                feet=c.robot_body_pos_w[:,q.feet];ground=c.ground_height(feet[:,:,:2])
                gravity=robot.data.projected_gravity_b[:,2].cpu().numpy()
                clearance=(root[:,2]-c.ground_height(root[:,:2])).cpu().numpy()
                goal=scene['tile']['goal']
                passed=(local[:,0]>=goal)&((feet[:,:,0]-c.tile_origins[c.tile,None,0]).amin(1).cpu().numpy()>=goal-.45)
                stable=((feet[:,:,2]-ground).abs().amax(1).cpu().numpy()<.18)&(gravity<-.85)&(abs(local[:,1])<1.5)&(clearance>.45)
                held=np.where(passed&stable,held+1,0)
                fallen=(gravity>-.4)|(clearance<.25)
                velocity=robot.data.root_link_lin_vel_w[:,:2].cpu().numpy()
                error+=np.linalg.norm(velocity-np.array([[x['speed'],0] for x in cases]),axis=1)*active;counts+=active
                terminal=active&((held>=25)|fallen|(step==round(args.duration/raw.step_dt)-1))
                for i in np.flatnonzero(terminal):
                    rows[i]=dict(case=cases[i],success=bool(held[i]>=25),fall=bool(fallen[i]),steps=step+1,time_s=(step+1)*raw.step_dt,progress_m=float(local[i,0]),speed_error_m_s=float(error[i]/counts[i]))
                active[terminal]=False
                if step%2==0:
                    for i in np.flatnonzero(before):
                        im=Image.new('RGB',(640,552),(18,23,32));im.paste(Image.fromarray(frames[i]),(0,30))
                        im.paste(Image.fromarray(rgbs[i]).resize((224,126)),(8,412))
                        draw=ImageDraw.Draw(im);case=cases[i]
                        draw.text((8,8),f'{scene["name"]} | {case["speed"]} m/s | {case["intervention"]} | t={step*.02:.2f}s',fill='white')
                        draw.text((8,394),'Policy RGB input',fill='white')
                        spec=memory.model.geometry
                        v=np.clip((predicted[i].reshape(spec.nx,spec.ny)-spec.z_min)/(spec.z_max-spec.z_min),0,1)
                        color=(np.stack((v,1-abs(2*v-1),1-v),-1)*255).astype(np.uint8).transpose(1,0,2)[::-1].copy()
                        im.paste(Image.fromarray(color).resize((192,126)),(240,412))
                        draw.text((240,394),'Predicted height',fill='white')
                        draw.text((442,430),'RGB-only perception',fill='white')
                        draw.text((442,450),'No depth input',fill='white')
                        writers[i].append_data(np.asarray(im))
                    drift=max(drift,float((raw.sim.data.cam_xpos[:,ci]-camera_pose[0]).abs().max()),float((raw.sim.data.cam_xmat[:,ci]-camera_pose[1]).abs().max()))
                if step%200==0:print(json.dumps(dict(step=step,active=int(active.sum()),results=rows)),flush=True)
                if not active.any():break
        arrays={k:np.stack([r[k] for r in trace]) for k in trace[0]}
        np.savez_compressed(out/'trace.npz',**arrays)
        for i,row in enumerate(rows):
            a={k:v[:row['steps'],i] for k,v in arrays.items()};support=a['stable_support'];margins=a['sole_edge_margin_m'][support]
            foot=a['foot_riser_force_n']>15;shin=a['shin_force_n']>15
            first=a['root_local'][:,0]<scene['tile']['start']+.45
            row.update(foot_impulse_ns=float(a['foot_riser_impulse_ns'].sum()),shin_impulse_ns=float(a['shin_impulse_ns'].sum()),first_step_collision=bool((foot[first]|shin[first]).any()),collision_free=not bool((foot|shin).any()),support_slip_speed_m_s=float(a['slip_distance_m'].sum()/max(a['support_seconds'].sum(),1e-8)),support_overhang_fraction=None if scene.get('heightfield') else float((margins<0).mean()) if len(margins) else None)
        audit=dict(black.audit,masked_worlds=[1,3],normal_worlds=[0,2],applications=step+1)
        (out/'mask_audit.json').write_text(json.dumps(audit,indent=2))
        report=dict(args=vars(args),scene=scene,checkpoint_sha256=file_hash(args.checkpoint),cases=rows,initial_state=initial,camera_max_pose_drift=drift,completed=sum(r['success'] for r in rows),total=4)
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:
        for writer in writers:writer.close()
        env.close()


if __name__=='__main__':main()
