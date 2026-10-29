"""Compare RGB geometry predictors on the SAME independently simulated trajectory.

Physics labels score predicted maps only. The selected learned policy controls
the robot; neither predictor receives the labels, positions, or decoded maps.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','checkpoint','reference_checkpoint','case_report','output'):
        p.add_argument('--'+k.replace('_','-'),required=True)
    p.add_argument('--case',type=int,required=True)
    p.add_argument('--duration',type=float,default=24.)
    p.add_argument('--stride',type=int,default=25)
    a=p.parse_args()
    if a.stride<1:p.error('Positive snapshot stride required')
    from .review import verify_review_worker
    verify_review_worker()
    import torch,numpy as np
    from .record_scene import build
    from .runtime import load_torch_control
    from .labels import PhysicsLabels
    from .task import floor_contact
    from .route_audit import RouteAudit
    from .targets import guard_failed,guard_diagnostic
    from ..hiking_gaussgym.runtime import RGBMemory
    from ..hiking_gaussgym.model import decode_height
    from ..hiking_gaussgym.voxel import heightmap_to_voxels
    from ..hiking_perception.geometry import grid_world,yaw_from_quat
    from ..velocity.recovery_terrain import profile_success
    torch.set_num_threads(2)
    settings=json.loads(Path(a.config).read_text())
    source=json.loads(Path(a.case_report).read_text())
    case=next(row['case'] for row in source['cases'] if row['case']['id']==a.case)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    env,settings,cases,tile=build(settings,case,source['seed'],a.duration,with_review_camera=False)
    try:
        obs,_=env.reset();raw=env.unwrapped;c=raw.command_manager.get_term('motion');robot=raw.scene['robot']
        actor,current=load_torch_control(a.checkpoint,obs,settings,raw.device)
        state=torch.load(a.reference_checkpoint,map_location='cpu',weights_only=False)
        reference=RGBMemory(state['infos']['gaussgym'],raw.num_envs,raw.device);del state
        if current.model.geometry!=reference.model.geometry:raise ValueError('Predictors need identical geometry grids')
        g=current.model.geometry;labels=PhysicsLabels(raw,g)
        reset=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device)
        route=RouteAudit(raw)
        xs=torch.linspace(g.x_min,g.x_max,g.nx,device=raw.device)
        ys=torch.linspace(g.y_min,g.y_max,g.ny,device=raw.device)
        xx,yy=torch.meshgrid(xs,ys,indexing='ij')
        roi=(xx>=.10)&(xx<=.90)&(yy.abs()<=.28)
        accum={};snapshots=[];input_rows=[];snapshot_steps=[];success=fall=guard=False

        def add(name,region,mask,pred,bins,truth,tgtbins,void,logits,occupancy):
            key=name+'/'+region
            v=accum.setdefault(key,dict(cells=0,correct_bins=0,clipped_abs_error_sum=0.,
                void_cells=0,void_correct_bins=0,void_positive_bias_sum=0.,
                saturated_cells=0,saturated_correct_bins=0,
                positive_bce_sum=0.,negative_bce_sum=0.,negative_count=0))
            saturated=(truth<=g.z_min)|(truth>=g.z_max)
            valid_void=mask&void;sat=mask&saturated
            v['cells']+=int(mask.sum());v['correct_bins']+=int(((bins==tgtbins)&mask).sum())
            error=pred-truth.clamp(g.z_min,g.z_max)
            v['clipped_abs_error_sum']+=float(error[mask].abs().sum())
            v['void_cells']+=int(valid_void.sum());v['void_correct_bins']+=int(((bins==tgtbins)&valid_void).sum())
            v['void_positive_bias_sum']+=float(error[valid_void].sum())
            v['saturated_cells']+=int(sat.sum());v['saturated_correct_bins']+=int(((bins==tgtbins)&sat).sum())
            positive=mask[...,None]&occupancy;negative=mask[...,None]&~occupancy
            v['positive_bce_sum']+=float(torch.nn.functional.softplus(-logits)[positive].sum())
            v['negative_bce_sum']+=float(torch.nn.functional.softplus(logits)[negative].sum())
            v['negative_count']+=int(negative.sum())

        with torch.inference_mode():
            for step in range(round(a.duration/raw.step_dt)):
                input_rows.append(dict(tokens=obs['kv_tokens']['img_tokens'][0].cpu().numpy().copy(),
                    proprio=obs['q_proprio'][0].cpu().numpy().copy(),reset=bool(reset[0])))
                z=current.step(obs,reset);old_z=reference.step(obs,reset)
                if step%a.stride==0:
                    target=labels.sample(torch.tensor([0],device=raw.device))
                    truth=target['height'][0].reshape(g.nx,g.ny)
                    valid=target['surface_valid'][0].reshape_as(truth)&torch.isfinite(truth)
                    occupancy,_=heightmap_to_voxels(truth[None],g.height_bins,g.z_min,g.z_max)
                    occupancy=occupancy[0].bool();tgtbins=occupancy.long().argmax(-1)
                    absolute=truth+robot.data.root_link_pos_w[0,2]
                    void=(absolute<.10)&valid
                    if tile['family']=='stair_gap':
                        xy=grid_world(robot.data.root_link_pos_w[:1],yaw_from_quat(robot.data.root_link_quat_w[:1]),g)[0]
                        x=xy[...,0].reshape_as(truth)-c.tile_origins[c.tile[0],0]
                        lo,hi=tile['gaps'][0];void&=(x>=lo)&(x<=hi)
                    elif tile['family']=='stairs':void[:]=False
                    row=dict(time_s=step*raw.step_dt,truth=truth.cpu().numpy(),void=void.cpu().numpy(),
                             valid=valid.cpu().numpy(),
                             rgb=raw.scene['head_cam'].data.rgb[0].cpu().numpy().copy())
                    for name,m,latent in [('current',current,z),('reference',reference,old_z)]:
                        logits,centroid,_,_=m.model.heads(latent[:1])
                        pred=decode_height(logits,centroid,g)[0];bins=logits[0].argmax(-1)
                        row[name]=pred.cpu().numpy()
                        for region,mask in [('all',valid),('front_foot',valid&roi)]:
                            add(name,region,mask,pred,bins,truth,tgtbins,void,logits[0],occupancy)
                            if step*raw.step_dt>=2.:
                                add(name,region+'_after_2s',mask,pred,bins,truth,tgtbins,void,logits[0],occupancy)
                    snapshots.append(row)
                    snapshot_steps.append(step)
                acted=obs.clone(False);acted['hiking_latent']=z
                action=actor(acted,stochastic_output=False)
                obs,_,done,_=env.step(action);reset=done.bool()
                if done.any():raise RuntimeError('Unexpected auto-reset')
                root=robot.data.root_link_pos_w
                fall=bool((robot.data.projected_gravity_b[0,2]>-.4)|((root[:,2]-c.ground_height(root[:,:2]))[0]<.25)|floor_contact(raw)[0])
                guard=bool(guard_failed(raw)[0])
                route.update(torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device))
                success=bool((profile_success(raw)&route.qualified)[0]) and not (fall or guard)
                if step%250==0:print(json.dumps(dict(step=step,snapshots=len(snapshots),fall=fall,success=success)),flush=True)
                if success or fall or guard:break
        metrics={}
        for key,v in accum.items():
            metrics[key]=dict(v,bin_accuracy=v['correct_bins']/max(v['cells'],1),
                clipped_height_mae_m=v['clipped_abs_error_sum']/max(v['cells'],1),
                void_bin_accuracy=v['void_correct_bins']/v['void_cells'] if v['void_cells'] else None,
                void_height_bias_m=v['void_positive_bias_sum']/v['void_cells'] if v['void_cells'] else None,
                saturated_bin_accuracy=v['saturated_correct_bins']/v['saturated_cells'] if v['saturated_cells'] else None,
                mean_positive_bce=v['positive_bce_sum']/max(v['cells'],1),
                mean_negative_bce=v['negative_bce_sum']/max(v['negative_count'],1))
        report=dict(checkpoint=str(Path(a.checkpoint).resolve()),reference_checkpoint=str(Path(a.reference_checkpoint).resolve()),
            case=case,seed=source['seed'],duration_s=(step+1)*raw.step_dt,success=success,fall=fall,
            actuator_guard_failed=guard,guard_diagnostic=guard_diagnostic(raw,0) if guard else None,
            shared_observed_trajectory=True,physics_truth_as_policy_input=False,decoder_used_for_control=False,
            rgb_update_hz=1/raw.step_dt,initial_state='training_walking_RSI',samples=len(snapshots),metrics=metrics,
            geometry=asdict(g),control_dt_s=raw.step_dt,
            replay_inputs=['inputs_tokens.npy','inputs_proprio.npy','inputs_reset.npy'],
            replay_labels_separate=True,
            note='Full-map diagnostic includes occluded/unseen cells. The two predictors see identical RGB/proprio histories; physics labels only score outputs.')
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        for key in ['truth','void','valid','rgb','current','reference']:
            np.save(out/(key+'.npy'),np.stack([r[key] for r in snapshots]))
        np.save(out/'times.npy',np.array([r['time_s'] for r in snapshots]))
        np.save(out/'snapshot_steps.npy',np.array(snapshot_steps,dtype=np.int64))
        for key in ['tokens','proprio','reset']:
            np.save(out/('inputs_'+key+'.npy'),np.stack([r[key] for r in input_rows]))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        chosen=np.linspace(0,len(snapshots)-1,min(4,len(snapshots))).astype(int)
        fig,axes=plt.subplots(len(chosen),4,figsize=(12,3*len(chosen)),squeeze=False)
        for i,index in enumerate(chosen):
            row=snapshots[index];axes[i,0].imshow(row['rgb']);axes[i,0].set_title(f'RGB at {row["time_s"]:.2f}s');axes[i,0].axis('off')
            for j,key in enumerate(['truth','reference','current'],1):
                axes[i,j].imshow(row[key].T,origin='lower',extent=[g.x_min,g.x_max,g.y_min,g.y_max],
                    vmin=g.z_min,vmax=.2,cmap='viridis',aspect='auto')
                axes[i,j].set_title(key+' relative height (m)');axes[i,j].set_xlabel('forward x (m)')
        fig.tight_layout();fig.savefig(out/'geometry_comparison.png',dpi=140);plt.close(fig)
        print(json.dumps({k:v for k,v in report.items() if k!='metrics'},indent=2),flush=True)
    finally:env.close()


if __name__=='__main__':main()
