"""Full physics-grid and per-route geometry evaluation; no learned edge classifier."""
import collections
import torch
from .model import decode_height
from .learning import autocast

@torch.no_grad()
def evaluate(model,data,plan,device,burn_in=16,stride=8,intervention='normal'):
    model.eval();groups=collections.defaultdict(lambda:dict(error=0.,cells=0,lv=0.,av=0.,frames=0));errors=[]
    for index in plan:
        b=data.batch([index],device)
        if intervention=='zero_rgb':b['tokens'].zero_()
        elif intervention=='zero_proprio':b['proprio'].zero_()
        elif intervention!='normal':raise ValueError('Unknown intervention')
        z,_=model.sequence(b['tokens'],b['proprio'],b['reset'])
        key='|'.join(data.metadata[index][0]);row=groups[key];g=model.geometry
        for t in range(burn_in,len(z),stride):
            with autocast(device):logits,offset,lv,av=model.heads(z[t])
            prediction=decode_height(logits.float(),offset.float(),g)
            truth=b['height'][t].reshape_as(prediction)
            valid=b['surface_valid'][t].reshape_as(truth)&b['valid'][t,:,None,None]&torch.isfinite(truth)&(truth>g.z_min)&(truth<g.z_max)
            error=(prediction-truth).abs()[valid].cpu();errors.append(error)
            row['error']+=float(error.sum());row['cells']+=len(error)
            row['lv']+=float((lv.float()-b['base_lin_vel_b'][t]).square().mean());row['av']+=float((av.float()-b['base_ang_vel_b'][t]).square().mean());row['frames']+=1
    populated=[v for v in groups.values() if v['cells']]
    if not populated:raise RuntimeError('Evaluation has no valid physics geometry')
    summary={k:dict(height_mae_m=v['error']/max(v['cells'],1),cells=v['cells'],
                    linvel_rmse_m_s=(v['lv']/max(v['frames'],1))**.5,angvel_rmse_rad_s=(v['av']/max(v['frames'],1))**.5)
             for k,v in groups.items()}
    error=torch.cat(errors)
    return dict(intervention=intervention,windows=len(plan),cells=len(error),height_mae_m=float(error.mean()),
                height_p95_m=float(torch.quantile(error,.95)),macro_height_mae_m=sum(v['error']/v['cells'] for v in populated)/len(populated),
                linvel_rmse_m_s=(sum(v['lv'] for v in groups.values())/sum(v['frames'] for v in groups.values()))**.5,
                angvel_rmse_rad_s=(sum(v['av'] for v in groups.values())/sum(v['frames'] for v in groups.values()))**.5,
                groups=summary,mask='all valid unsaturated physics terrain; not camera visibility')
