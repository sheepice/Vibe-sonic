# SONIC + ViBe online-reference velocity task

This is an independent post-training task. Original task modules, vendor code,
datasets and checkpoints are not modified. It starts from the existing ViBe
`model_14999.pt`; the original SONIC encoder, quantizer and base decoder remain
frozen. Trainable modules are the existing rank-16 decoder LoRA, visual attention
extractor, and a new zero-initialized explicit-command encoder. The critic is
initialized afresh because the reward and task distribution change.

## Task and data

Each rank uses its existing `data/full/rank_N` shard. Fixed environment groups:

- 60%: online SONIC planner reference + independently supervised planar velocity
  and heading + randomly sampled procedural terrain.
- 20%: original GRAIL stair motion and its paired geometry.
- 20%: original flat mixed sources, including SONIC, LaFAN, Style100, Real Pico,
  using the existing source weights. A small Gaussian policy-prior loss toward
  original SONIC actions is applied only to these flat samples.

Five CPU ONNX rolling streams per rank (idle, 0.25/0.4/0.55/0.7 m/s) are replanned
at 5 Hz from generated context. Physics states, resets, terrain and phase offsets
remain independent per environment. This is online reference generation, not
replaying precomputed stair trajectories. Sharing amortizes the planner's fixed
batch-one interface. References always retain the 0.9-second tokenizer horizon.
Idle uses SONIC mode 0: `target_vel=0` in walk mode would incorrectly mean the
default walking speed. Small independent speed changes are sampled every 4 s.

The current scope is +X forward walking and stopping, with randomized initial
heading error and terrain approach. Arbitrary turning and large stop/restart
transitions within an episode are not yet trained. A deployer must supply the new
8-dimensional `velocity_command` group; the original ViBe ONNX exporter refuses
this new actor to prevent silently dropping command conditioning.

## Learning objectives

Original tracking objectives and tracking terminations apply to replay groups.
Online tasks supervise horizontal velocity, heading, upright posture, upper-body
style, forward progress and foot slip. Root height is free to rise/fall with the
terrain. Failure uses actual tilt, root-to-ground clearance and lane departure.
Success requires passing the full staircase upright with feet near its surface.

A weak BRACE-inspired foot-height target lifts the nominal foot clearance onto
the local surface, used only for rewards. Terrain-adjusted poses do not enter the
SONIC tokenizer. P-BFM motivates separating behavior from commanded horizontal
transport; this remains PPO adaptation of SONIC, not an implementation of FB RL.

The actor receives RGB 112x63 through frozen Theia, proprioception, the original
SONIC reference window and explicit command. No depth or height map enters the
actor. The critic and reward computation may use privileged terrain. Original
domain randomization is retained. Procedural stairs cover ascent, descent,
ascent-descent and flat terrain; 4-18 cm riser levels progressively unlock.

## Launch

```bash
cd /mnt/workspace/wjw/GRAIL/vibe_grail_joint
.venv/bin/python -m vibe_grail_joint.velocity.launch --gpus 0 \
  --config configs/online_velocity.json --log-dir records/velocity_smoke \
  --num-envs 20 --iterations 8 --smoke

.venv/bin/python -m vibe_grail_joint.velocity.launch --gpus 0,1,2,3,4,5,6,7 \
  --config configs/online_velocity.json --log-dir runs/NEW_UNIQUE_DIRECTORY \
  --num-envs 4090 --iterations 6000
```

Environment count is a multiple of ten for exact 60/20/20 proportions. PPO
minibatches are capped near 1024 samples/rank without shrinking the model. Run
measured memory smoke checks before increasing environment count. The launcher
refuses GPUs occupied by another job; it never signals existing training.

Check `rank_N_status.json` and `rank_N_metrics.jsonl` for update count, losses,
per-task rewards, speed error, success/fall counters and peak CUDA allocation.
Checkpoints are written atomically. Frozen-base hashes are checked after a run.
Curriculum level and randomized geometry are in the task configuration; evaluation
must use new terrain seeds and separately report complete-stair success and flat
motion retention. Passing the smoke test is not a claim of learned stair success.
