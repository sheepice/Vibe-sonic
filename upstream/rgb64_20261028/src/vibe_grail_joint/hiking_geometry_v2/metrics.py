"""Fixed, explicitly stratified validation and validation-only edge calibration."""
import collections
import torch
from ..hiking_perception.geometry import decode_height, height_bins


def evaluate(model, dataset, plan, batch_size, device, burn_in, recon_stride,
             mode='normal', calibrated_threshold=.5):
    thresholds=sorted(set([round(i/10,1) for i in range(1,10)]+[float(calibrated_threshold)]))
    group_values=collections.defaultdict(list)
    group_counts=collections.defaultdict(lambda:torch.zeros(len(thresholds),4,dtype=torch.long))
    historical=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(plan),batch_size):
            indices=plan[start:start+batch_size]
            b=dataset.batch(indices,device)
            if mode=='zero_image': b['tokens'].zero_()
            elif mode=='zero_proprio': b['proprio'].zero_()
            elif mode!='normal': raise ValueError(mode)
            h,_=model.sequence(b['tokens'],b['proprio'],b['age'],b['reset'])
            sl=slice(burn_in,None,recon_stride)
            target=b['height'][sl]
            logits,offset,edge=model.heads(h[sl].flatten(0,1))
            error=(decode_height(logits,offset,model.geometry).reshape_as(target)-target).abs()
            visible=b['visible'][sl] & height_bins(target,model.geometry)[2] & b['valid'][sl,:,None]
            history=b['historical_visible'][sl] & height_bins(target,model.geometry)[2] & b['valid'][sl,:,None]
            historical.append(error[history].cpu())
            probability=edge.sigmoid().reshape_as(target)
            truth=b['edge'][sl].bool()
            for j,index in enumerate(indices):
                key='|'.join(dataset.metadata[index][0])
                group_values[key].append(error[:,j][visible[:,j]].cpu())
                true=truth[:,j]; mask=visible[:,j]
                for ti,threshold in enumerate(thresholds):
                    pred=probability[:,j]>threshold
                    group_counts[key][ti]+=torch.tensor([int((pred&true&mask).sum()),int((pred&~true&mask).sum()),
                                                        int((~pred&true&mask).sum()),int((~pred&~true&mask).sum())])
    def summarize_counts(counts):
        tp,fp,fn,tn=counts.tolist()
        precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
        return dict(confusion_tp_fp_fn_tn=[tp,fp,fn,tn],precision=precision,recall=recall,
                    f1=2*tp/max(2*tp+fp+fn,1),f05=1.25*tp/max(1.25*tp+fp+.25*fn,1))
    groups={}; arrays=[]; total_counts=torch.zeros(len(thresholds),4,dtype=torch.long)
    for key,chunks in group_values.items():
        values=torch.cat(chunks); arrays.append(values); total_counts+=group_counts[key]
        groups[key]=dict(valid_cells=len(values),height_mae_m=float(values.mean()) if len(values) else None,
                         edge_at_05=summarize_counts(group_counts[key][thresholds.index(.5)]))
    values=torch.cat(arrays)
    if not len(values): raise RuntimeError('Validation has no visible valid cells')
    history_values=torch.cat(historical)
    pr={str(t):summarize_counts(total_counts[i]) for i,t in enumerate(thresholds)}
    macro_mae=sum(v['height_mae_m'] for v in groups.values() if v['valid_cells'])/sum(bool(v['valid_cells']) for v in groups.values())
    supported=[v['edge_at_05']['f1'] for v in groups.values() if sum(v['edge_at_05']['confusion_tp_fp_fn_tn'][::2])>0]
    macro_f1=sum(supported)/max(len(supported),1)
    return dict(mode=mode,windows=len(plan),groups=groups,valid_cells=len(values),
                height_mae_m=float(values.mean()),height_p95_m=float(values.quantile(.95)),
                macro_height_mae_m=macro_mae,macro_edge_f1_at_05=macro_f1,
                selection_score=macro_mae+.05*(1-macro_f1),edge_thresholds=pr,
                historical_cells=len(history_values),historical_height_mae_m=float(history_values.mean()) if len(history_values) else None)


def calibrate(validation):
    # Selection uses validation only; no assertion that this threshold is safe.
    return float(max(validation['edge_thresholds'],key=lambda t:validation['edge_thresholds'][t]['f05']))
