"""Train geometry probes, recurrent perception or a local action world model."""
import argparse
from dataclasses import replace
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', nargs='+', required=True); p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=('probe','geometry','world','joint'), required=True)
    p.add_argument('--action-mode', choices=('normal','zero'), default='normal')
    p.add_argument('--collision-fraction', type=float, default=0.,help='Positive-window sampling for world/joint training; validation keeps natural frequencies')
    p.add_argument('--history-weight',type=float,default=.25,help='Weight of previously visible, currently occluded terrain supervision')
    p.add_argument('--geometry-checkpoint'); p.add_argument('--resume')
    p.add_argument('--updates', type=int, default=5000); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--sequence', type=int, default=32); p.add_argument('--burn-in', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--seed', type=int, default=137)
    p.add_argument('--eval-every', type=int, default=100); p.add_argument('--device', choices=('cpu','cuda:0'), default='cuda:0')
    p.add_argument('--validation-batches',type=int,default=4)
    args = p.parse_args()
    if args.device.startswith('cuda'):
        from ..gpu_guard import verify_guarded_worker
        verify_guarded_worker()
    import torch
    from .contracts import contract, parse_contract, write_json
    from .models import GeometryMemory, ActionWorldModel, geometry_loss, world_loss
    from .dataset import SequenceDataset
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    if args.burn_in < 0 or args.sequence <= args.burn_in or args.updates < 1 or args.history_weight < 0 or args.validation_batches < 1:
        p.error('Require updates>0 and sequence>burn-in>=0')
    training_contract=dict(sequence=args.sequence,burn_in=args.burn_in,action_mode=args.action_mode,
        collision_fraction=args.collision_fraction,history_weight=args.history_weight,lr=args.lr,sampling='shard_local_v2',validation_batches=args.validation_batches)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    train = SequenceDataset(args.data, args.sequence, 'train')
    val = SequenceDataset(args.data, args.sequence, 'val')
    if set(train.scenes) & set(val.scenes):
        raise RuntimeError('Scene leakage')
    geometry, model_spec = parse_contract(train.contract)
    model_spec = replace(model_spec, temporal=args.mode != 'probe')
    encoder = GeometryMemory(geometry, model_spec).to(args.device)
    world = None
    if args.geometry_checkpoint and args.mode == 'geometry':
        prior = torch.load(args.geometry_checkpoint, map_location=args.device, weights_only=False)
        if parse_contract(prior['contract']) != (geometry, model_spec):
            raise ValueError('Perception continuation contract mismatch')
        encoder.load_state_dict(prior['encoder'])
    if args.mode in ('world', 'joint'):
        if not args.geometry_checkpoint:
            p.error('--geometry-checkpoint is required for world training')
        state = torch.load(args.geometry_checkpoint, map_location=args.device, weights_only=False)
        geometry, model_spec = parse_contract(state['contract'])
        if parse_contract(train.contract)[0] != geometry:
            raise ValueError('Dataset and encoder geometry mismatch')
        encoder = GeometryMemory(geometry, model_spec).to(args.device)
        encoder.load_state_dict(state['encoder'])
        if args.mode == 'world': encoder.requires_grad_(False).eval()
        world = ActionWorldModel(model_spec).to(args.device)
        if args.sequence < args.burn_in+max(model_spec.horizon_steps):
            p.error('World sequence must include burn-in plus all action horizons')
    optimized = torch.nn.ModuleList([encoder,world]) if args.mode == 'joint' else world if world is not None else encoder
    optimizer = torch.optim.AdamW(optimized.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed+1)
    start, best = 0, float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        if state.get('training_contract') != training_contract:
            raise ValueError('Resume objective/sampling contract mismatch or legacy checkpoint; use --geometry-checkpoint for explicit encoder warm start')
        if state['mode'] != args.mode or state['dataset_fingerprint'] != train.fingerprint or state.get('action_mode','normal') != args.action_mode:
            raise ValueError('Resume mode/dataset mismatch')
        if state['contract'] != contract(geometry, model_spec):
            # JSON contracts contain lists, in-memory checkpoint contracts may use tuples.
            if parse_contract(state['contract']) != (geometry, model_spec): raise ValueError('Resume model contract mismatch')
        encoder.load_state_dict(state['encoder'])
        if world is not None:
            world.load_state_dict(state['world'])
        optimizer.load_state_dict(state['optimizer']); generator.set_state(state['generator'].cpu())
        if 'validation_generator' in state: val_generator.set_state(state['validation_generator'].cpu())
        start, best = state['update']+1, state['best_validation']

    def objective(batch, shuffle_actions=False):
        h, _ = encoder.sequence(batch['tokens'], batch['proprio'], batch['age'], batch['reset'])
        def terrain_objective():
            sl=slice(args.burn_in,None)
            targets={k:batch[k][sl].flatten(0,1) for k in ('height','edge','visible')}
            targets['visible'] &= batch['valid'][sl].flatten()[:,None]
            loss,metrics=geometry_loss(encoder,h[sl].flatten(0,1),targets)
            if 'historical_visible' in batch and args.history_weight and batch['historical_visible'].any():
                targets['visible']=batch['historical_visible'][sl].flatten(0,1)&batch['valid'][sl].flatten()[:,None]
                hl,hm=geometry_loss(encoder,h[sl].flatten(0,1),targets)
                loss=loss+args.history_weight*hl
                metrics.update(history_height_mae_m=hm['height_mae_m'],history_cells=hm['visible_cells'])
            return loss,metrics
        if world is None:
            return terrain_objective()
        n = max(model_spec.horizon_steps); first = args.burn_in
        future = {k: batch[k][first:first+n] for k in ('action','command','reference_twist','next_state','contact','done')}
        if shuffle_actions:
            future['action'] = future['action'].flip(1)
        if args.action_mode == 'zero': future['action'] = torch.zeros_like(future['action'])
        loss, metrics = world_loss(world, h[first], future)
        if args.mode == 'joint':
            geom_loss, gm = terrain_objective()
            loss = geom_loss + .2*loss
            metrics.update(gm, joint_loss=float(loss.detach()))
        return loss, metrics

    write_json(out/'experiment.json', dict(args=vars(args), contract=contract(geometry,model_spec),
        dataset_fingerprint=train.fingerprint, train_windows=len(train.windows), val_windows=len(val.windows),
        train_scenes=train.scenes, val_scenes=val.scenes, prediction_only=True,
        note='A low supervised loss does not establish locomotion improvement'))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(out/'tensorboard'))
    try:
        for update in range(start, start+args.updates):
            optimized.train()
            batch = train.sample(args.batch_size, generator, args.device, collision_fraction=args.collision_fraction)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = objective(batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite supervised loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(optimized.parameters(), 1.); optimizer.step()
            for key,value in metrics.items():
                writer.add_scalar('train/'+key, value, update)
            if update % args.eval_every == 0 or update == start+args.updates-1:
                optimized.eval()
                # Use the same validation windows for each checkpoint selection,
                # rather than rewarding an easier random batch on a later update.
                val_generator.manual_seed(args.seed+1)
                with torch.no_grad():
                    val_values=[]; metric_rows=[]
                    for _ in range(args.validation_batches):
                        vb = val.sample(args.batch_size, val_generator, args.device)
                        vl, row = objective(vb)
                        if world is not None:
                            _, shuffled = objective(vb, True)
                            row['action_shuffled_loss'] = shuffled['world_loss']
                        val_values.append(vl); metric_rows.append(row)
                    val_loss=torch.stack(val_values).mean()
                    vm={key:sum(row.get(key,0.) for row in metric_rows)/len(metric_rows) for key in set().union(*(row.keys() for row in metric_rows))}
                improved = float(val_loss) < best; best = min(best,float(val_loss))
                for key,value in vm.items():
                    writer.add_scalar('validation/'+key, value, update)
                checkpoint = dict(schema=1, mode=args.mode, contract=contract(geometry,model_spec),
                    scene_key_version=2,train_scenes=train.scenes,val_scenes=val.scenes,
                    encoder=encoder.state_dict(), world=world.state_dict() if world is not None else None,
                    optimizer=optimizer.state_dict(), update=update, best_validation=best,
                    generator=generator.get_state(), validation_generator=val_generator.get_state(),
                    action_mode=args.action_mode, training_contract=training_contract, dataset_fingerprint=train.fingerprint)
                tmp = out/'last.pt.tmp'; torch.save(checkpoint,tmp); tmp.replace(out/'last.pt')
                if improved:
                    tmp = out/'best.pt.tmp'; torch.save(checkpoint,tmp); tmp.replace(out/'best.pt')
                report = dict(update=update, train=metrics, validation=vm, best_validation=best)
                write_json(out/'status.json',report)
                with (out/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(report)+'\n')
                print(json.dumps(report),flush=True)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
