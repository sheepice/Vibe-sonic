"""Score saved RGB/proprio sequences on CPU; labels never enter the predictor.

This is a fixed-input perception diagnostic, NOT a robot-control evaluation.
Every compared checkpoint receives exactly the same causal sensor history.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',required=True)
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    import numpy as np
    import torch
    from ..hiking_gaussgym.model import load_model,decode_height
    from ..hiking_gaussgym.voxel import heightmap_to_voxels
    torch.set_num_threads(2)
    source=Path(a.trajectory);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    original=json.loads((source/'report.json').read_text())
    inputs={k:torch.from_numpy(np.load(source/('inputs_'+k+'.npy'))) for k in ('tokens','proprio','reset')}
    steps=np.load(source/'snapshot_steps.npy');times=torch.from_numpy(np.load(source/'times.npy'))
    truth=torch.from_numpy(np.load(source/'truth.npy'))
    valid=torch.from_numpy(np.load(source/'valid.npy')).bool() & torch.isfinite(truth)
    void=torch.from_numpy(np.load(source/'void.npy')).bool() & valid
    assert inputs['tokens'].shape[1:]==(21,192) and inputs['proprio'].shape==(len(inputs['tokens']),64)
    assert inputs['reset'].shape==(len(inputs['tokens']),) and len(steps)==len(truth)==len(times)
    assert len(set(steps.tolist()))==len(steps) and min(steps)>=0 and max(steps)<len(inputs['tokens'])
    assert torch.isfinite(inputs['tokens']).all() and torch.isfinite(inputs['proprio']).all()
    report=dict(trajectory=str(source.resolve()),same_inputs_for_all_checkpoints=True,
        robot_control_evaluation=False,physics_labels_as_predictor_input=False,results=[],
        input_sha256={k:hashlib.sha256((source/('inputs_'+k+'.npy')).read_bytes()).hexdigest() for k in inputs},
        trajectory_driver=original['checkpoint'],trajectory_initial_state=original['initial_state'],
        rgb_update_hz=original['rgb_update_hz'],note='Offline CPU diagnostic on one recorded policy trajectory; full maps include occluded cells.')
    heights=[]
    for number,path in enumerate(a.checkpoints):
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=load_model(state['infos']['gaussgym'],'cpu').eval().requires_grad_(False)
        del state
        g=model.geometry
        if asdict(g)!=original['geometry']:raise ValueError('Geometry grid mismatch')
        h,c=model.initial(1,'cpu');predictions=[];classes=[];wanted=set(steps.tolist())
        with torch.inference_mode():
            for t in range(len(inputs['tokens'])):
                # Deliberately enumerate the only encoder inputs; labels, robot
                # position, scene identity and decoded maps are not arguments.
                latent,h,c=model.step(inputs['tokens'][t:t+1],inputs['proprio'][t:t+1],h,c,inputs['reset'][t:t+1])
                if t in wanted:
                    logits,centroid,_,_=model.heads(latent)
                    predictions.append(decode_height(logits,centroid,g)[0])
                    classes.append(logits[0].argmax(-1))
        predicted=torch.stack(predictions);bins=torch.stack(classes)
        occupancy,_=heightmap_to_voxels(truth,g.height_bins,g.z_min,g.z_max);targets=occupancy.long().argmax(-1)
        x=torch.linspace(g.x_min,g.x_max,g.nx);y=torch.linspace(g.y_min,g.y_max,g.ny)
        xx,yy=torch.meshgrid(x,y,indexing='ij');roi=(xx>=.10)&(xx<=.90)&(yy.abs()<=.28)
        def score(mask):
            gap=mask&void;low=mask&(targets==0);low_pred=mask&(bins==0)
            tp=int((low&low_pred).sum());n=int(mask.sum());ng=int(gap.sum())
            error=predicted-truth.clamp(g.z_min,g.z_max)
            return dict(cells=n,bin_accuracy=float((bins[mask]==targets[mask]).float().mean()) if n else None,
                clipped_height_mae_m=float(error[mask].abs().mean()) if n else None,
                void_cells=ng,void_bin_accuracy=float((bins[gap]==targets[gap]).float().mean()) if ng else None,
                void_height_bias_m=float(error[gap].mean()) if ng else None,
                lower_saturated_cells=int(low.sum()),predicted_lower_cells=int(low_pred.sum()),
                lower_saturation_recall=tp/int(low.sum()) if low.any() else None,
                lower_saturation_precision=tp/int(low_pred.sum()) if low_pred.any() else None)
        result=dict(checkpoint=str(Path(path).resolve()),
            metrics={name:score(mask) for name,mask in [('all',valid),('front_foot',valid&roi),
                ('front_foot_after_2s',valid&roi&(times>=2.)[:,None,None])]},
            output_height_file=f'height_{number}.npy',output_bin_file=f'bin_{number}.npy')
        for label,key in [('current','checkpoint'),('reference','reference_checkpoint')]:
            if Path(path).resolve()==Path(original[key]).resolve():
                reference=np.load(source/(label+'.npy'))
                difference=float(np.abs(predicted.numpy()-reference).max())
                result['in_simulation_prediction_max_abs_difference_m']=difference
                # A few near-tied class decisions can differ across CPU/CUDA;
                # report mismatched cells rather than silently smoothing them.
                result['in_simulation_prediction_mae_m']=float(np.abs(predicted.numpy()-reference).mean())
        np.save(out/result['output_height_file'],predicted.numpy());np.save(out/result['output_bin_file'],bins.numpy())
        report['results'].append(result);heights.append(predicted.numpy())
        print(json.dumps(result),flush=True);del model
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plot the two snapshots with the most front-region gaps after warmup.
    counts=(void&roi&(times>=2.)[:,None,None]).sum((1,2)).numpy()
    chosen=np.sort(np.argsort(counts)[-min(2,len(counts)):])
    fig,axes=plt.subplots(len(chosen),len(heights)+2,figsize=(3*(len(heights)+2),3*len(chosen)),squeeze=False)
    rgb=np.load(source/'rgb.npy')
    for row,index in enumerate(chosen):
        axes[row,0].imshow(rgb[index]);axes[row,0].set_title(f'RGB {float(times[index]):.2f}s');axes[row,0].axis('off')
        for col,(title,values) in enumerate([('truth',truth.numpy()),*[(Path(path).parent.name.replace('stones_','').replace('_20261025','')+'\n'+Path(path).stem,values) for path,values in zip(a.checkpoints,heights)]],1):
            axes[row,col].imshow(values[index].T,origin='lower',extent=[g.x_min,g.x_max,g.y_min,g.y_max],vmin=g.z_min,vmax=-.4,cmap='viridis',aspect='auto')
            axes[row,col].set_title(title,fontsize=8);axes[row,col].set_xlabel('forward x (m)')
    fig.tight_layout();fig.savefig(out/'fixed_input_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
