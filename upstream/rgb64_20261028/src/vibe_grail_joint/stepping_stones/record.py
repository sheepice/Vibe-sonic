"""Write only a combined comparison video; actual RGB panels and fixed camera."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','output','case-report'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--record-case',type=int,required=True)
    p.add_argument('--comparison',choices=('baseline','black_rgb'),default='baseline')
    p.add_argument('--baseline')
    p.add_argument('--duration',type=float,default=40.)
    args=p.parse_args()
    if args.comparison=='baseline' and not args.baseline:p.error('--baseline is required')
    from .review import verify_review_worker
    verify_review_worker()
    import mjlab,torch,numpy as np,imageio.v2 as iio
    from PIL import Image,ImageDraw,ImageFont
    from .record_scene import build
    from .runtime import load_torch_control
    from .task import floor_contact
    from .route_audit import RouteAudit
    from ..velocity.recovery_terrain import profile_success
    from ..hiking_gaussgym.black_rgb import BlackRGB
    from ..hiking_perception.control import file_hash
    torch.set_num_threads(2)
    source=json.loads(Path(args.case_report).read_text())
    case=next(row['case'] for row in source['cases'] if row['case']['id']==args.record_case)
    seed=source['seed'];torch.manual_seed(seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    settings=json.loads(Path(args.config).read_text())
    env,settings,cases,tile=build(settings,case,seed,args.duration)
    writer=None
    try:
        obs,_=env.reset();raw=env.unwrapped;command=raw.command_manager.get_term('motion');robot=raw.scene['robot']
        actor,memory=load_torch_control(args.checkpoint,obs,settings,raw.device)
        baseline=baseline_memory=None
        if args.comparison=='baseline':baseline,baseline_memory=load_torch_control(args.baseline,obs,settings,raw.device)
        black=BlackRGB(raw) if args.comparison=='black_rgb' else None
        left_name='Stair baseline | '+Path(args.baseline).stem if baseline else 'Stones policy | RGB masked'
        right_name='Stones policy | '+Path(args.checkpoint).stem
        initial={k:getattr(robot.data,k).cpu().tolist() for k in
                 ('root_link_pos_w','root_link_quat_w','root_link_lin_vel_w','root_link_ang_vel_w','joint_pos','joint_vel')}
        initial_error=max(float(np.max(np.abs(np.array(v[0])-np.array(v[1])))) for v in initial.values())
        if initial_error>1e-7:raise RuntimeError(f'Mismatched initial states: {initial_error}')
        camera=raw.scene['review_camera'];ci=camera.camera_idx
        if int(raw.sim.mj_model.cam_bodyid[ci])!=0:raise RuntimeError('Review camera is not world-fixed')
        pose=(raw.sim.data.cam_xpos[:,ci].clone(),raw.sim.data.cam_xmat[:,ci].clone())
        drift=0.;active=torch.ones(2,dtype=torch.bool,device=raw.device)
        reset=torch.ones_like(active);rows=[None,None];last=[None,None]
        route=RouteAudit(raw)
        sums=torch.zeros(2,4,device=raw.device);counts=torch.zeros(2,device=raw.device)
        same_top=torch.zeros(2,device=raw.device);trace=[]
        font=ImageFont.load_default(size=21);small=ImageFont.load_default(size=17)
        writer=iio.get_writer(out/'comparison_fixed_rgb.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1,
                             ffmpeg_params=['-threads','1','-pix_fmt','yuv420p'])

        def panel(world,frame,rgb,t,result):
            image=Image.new('RGB',(960,816),(18,23,32));image.paste(Image.fromarray(frame),(0,40))
            image.paste(Image.fromarray(rgb).resize((336,189)),(12,612))
            draw=ImageDraw.Draw(image);title=left_name if world==0 else right_name
            draw.text((12,9),title,font=font,fill='white')
            draw.text((12,588),'Policy RGB input (112 x 63)',font=small,fill='white')
            label='RUNNING' if result is None else 'SUCCESS' if result['success'] else result['reason'].upper()+' | frame held'
            color=(235,240,255) if result is None else (92,228,140) if result['success'] else (255,120,100)
            draw.text((368,612),label,font=font,fill=color)
            draw.text((368,649),f"Command: {case['speed']:.2f} m/s | time: {t:.2f} s",font=small,fill='white')
            draw.text((368,681),f"{case['family']} | level {case['level']} | seed {seed}",font=small,fill='white')
            draw.text((368,713),f"Nominal gap: {100*tile.get('gap_m',0):.1f} cm",font=small,fill='white')
            draw.text((368,745),'Fixed observer; RGB-only policy; no depth input',font=small,fill='white')
            draw.text((368,777),'Selected demonstration; all-case scores are in the report',font=small,fill=(170,185,200))
            return np.asarray(image)

        with torch.inference_mode():
            for step in range(round(args.duration/raw.step_dt)):
                acted=obs.clone()
                if black:
                    acted['kv_tokens','img_tokens'][0].copy_(black.tokens[0]);acted['q_cls'][0].copy_(black.cls[0])
                acted['hiking_latent']=memory.step(acted,reset)
                actions=actor(acted,stochastic_output=False)
                if baseline:
                    base_obs=obs.clone(False);base_obs['hiking_latent']=baseline_memory.step(obs,reset)
                    actions[0]=baseline(base_obs,stochastic_output=False)[0]
                if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite action in recorded trial')
                before=active.clone()
                obs,_,done,_=env.step(actions);reset=done.bool()
                if done.any():raise RuntimeError('Unexpected auto-reset')
                root=robot.data.root_link_pos_w;local=root-command.tile_origins[command.tile]
                clearance=root[:,2]-command.ground_height(root[:,:2])
                floor=floor_contact(raw)
                fallen=(robot.data.projected_gravity_b[:,2]>-.4)|(clearance<.25)|floor
                route.update(before)
                from .targets import guard_failed
                guard=guard_failed(raw)
                success=profile_success(raw)&~fallen&route.qualified&~guard
                quality=raw.quality.step_data
                sums[:,0]+=(robot.data.root_link_lin_vel_w[:,0]-command.speed).abs()*before
                sums[:,1]+=quality['foot_riser_impulse_ns'].sum(-1)*before
                sums[:,2]+=quality['shin_impulse_ns'].sum(-1)*before
                sums[:,3]+=quality['slip_distance_m'].sum(-1)*before;counts+=before
                feet=command.robot_body_pos_w[:,[3,6]]
                surface=command.surface_ids(feet[:,:,:2],torch.arange(2,device=raw.device))
                same=(surface[:,0]==surface[:,1])&(surface[:,0]>=0)&raw.quality.sensor_support.all(-1)
                same_top+=same*before
                terminal=before&(fallen|success|guard|(step==round(args.duration/raw.step_dt)-1))
                for i in terminal.nonzero().flatten().tolist():
                    reason='actuator guard' if guard[i] else 'success' if success[i] else 'floor contact' if floor[i] else 'fall' if fallen[i] else 'timeout'
                    rows[i]=dict(case=cases[i],success=bool(success[i]),fall=bool(fallen[i]),reason=reason,
                                 time_s=(step+1)*raw.step_dt,progress_m=float(local[i,0]),
                                 speed_error_m_s=float(sums[i,0]/counts[i]),foot_impulse_ns=float(sums[i,1]),
                                 shin_impulse_ns=float(sums[i,2]),slip_distance_m=float(sums[i,3]),
                                 both_feet_same_surface_s=float(same_top[i]*raw.step_dt),**route.row(i))
                active[terminal]=False
                if step%2==1 or terminal.any():
                    frames=camera.data.rgb.cpu().numpy().copy();rgbs=raw.scene['head_cam'].data.rgb.cpu().numpy().copy()
                    if black:rgbs[0]=0
                    for i in before.nonzero().flatten().tolist():last[i]=panel(i,frames[i],rgbs[i],(step+1)*raw.step_dt,rows[i])
                    drift=max(drift,float((raw.sim.data.cam_xpos[:,ci]-pose[0]).abs().max()),float((raw.sim.data.cam_xmat[:,ci]-pose[1]).abs().max()))
                    if drift>1e-7:raise RuntimeError(f'Observer camera moved: {drift}')
                    if step%2==1 or not active.any():writer.append_data(np.concatenate(last,axis=1))
                if step%5==0 or terminal.any():
                    trace.append(dict(t=(step+1)*raw.step_dt,root=local.cpu().tolist(),feet=(feet-command.tile_origins[command.tile,None]).cpu().tolist(),
                                      support=raw.quality.sensor_support.cpu().tolist(),surface=surface.cpu().tolist(),active=before.cpu().tolist()))
                if step%250==0:print(json.dumps(dict(step=step,active=int(active.sum()),results=rows)),flush=True)
                if not active.any():break
        final=Image.fromarray(np.concatenate(last,axis=1));draw=ImageDraw.Draw(final)
        draw.rectangle((0,0,1920,38),fill=(18,23,32));draw.text((16,8),'RESULT HOLD | both trials finished',font=font,fill='white')
        for _ in range(25):writer.append_data(np.asarray(final))
        report=dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=file_hash(args.checkpoint),
                    baseline=str(Path(args.baseline).resolve()) if args.baseline else None,comparison=args.comparison,
                    seed=seed,case=case,terrain=tile,cases=rows,initial_state=initial,initial_state_max_abs_error=initial_error,
                    camera_max_pose_drift=drift,camera_parent='world',camera_position=pose[0].cpu().tolist(),
                    video=str((out/'comparison_fixed_rgb.mp4').resolve()),only_combined_video=True,policy_rgb_included=True,
                    selected_demonstration=True,source_report=str(Path(args.case_report).resolve()),
                    success_definition='full route, both feet on far landing, upright for 0.5 s, no gap-floor contact; ordered upper platform contacts for stair-gap courses',
                    trace=trace,black_rgb_audit=black.audit if black else None)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('trace','initial_state','terrain')},indent=2),flush=True)
    finally:
        if writer:writer.close()
        env.close()


if __name__=='__main__':main()
