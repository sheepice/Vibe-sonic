# G1 策略录制与注意力回放

当前代码已在本机和 G1 安装 `rosbags==0.11.9`，无需启动 ROS 节点。文件是标准 ROS 2 SQLite3 bag（`metadata.yaml` 和 `.db3`），不是 ROS 1 单文件 `.bag`。[格式与工具说明](https://ternaris.gitlab.io/rosbags/topics/rosbag2.html)

## 1. G1：启动策略时一起录制

先按 [G1_RUN_COMMANDS_ZH.md](G1_RUN_COMMANDS_ZH.md) 完成频率设置、相机检查和现场支撑。以下运动命令由你运行，助手没有执行它。

```bash
cd /home/unitree/vibe_g1
VIBE_TRIAL="$PWD/records/trial_$(date +%Y%m%d_%H%M%S)"

taskset -c 2,3 bash hardware/deployment/g1_python.sh \
  -m vibe_grail_joint.deploy.run \
  --bundle models/vggt_hierarchical_rgb64 \
  --interface enP8p1s0 \
  --mode control --execute --release-sport-mode --input remote \
  --provider tensorrt \
  --engine-dir engines/vggt_hierarchical_rgb64_trt10_fp32 \
  --inference-cpus 4,5,6,7 \
  --serial 254522074774 --allow-unmeasured-camera \
  --speed 0.4 --turn-rate 0.45 --backward-speed 0.1 --seconds 120 \
  --log "$VIBE_TRIAL/control.jsonl" \
  --record-bag "$VIBE_TRIAL/bag" --record-raw-hz 10 --record-max-mib 4096
```

说明：唯一新增的功能参数是最后一行。开始显示 `BAG RECORDING` 后，相机启动即录制，覆盖等待初始化、姿态保持和策略阶段。R1＋START ≥1 秒初始化，姿态就绪且双脚落地、有支撑后，松开全部按键，再单独 A ≥1 秒启动策略；方向键前后／转向，B 停步并继续策略。按键流程未改动。

说明：紧急情况按 SELECT 锁存软件阻尼并让吊具承托，支撑好后 Y ≥2 秒退出。SELECT 后程序可能继续保持阻尼，必须正常退出才能完整关闭 bag。不要用拔电、断 SSH 或 `kill -9` 收尾；它们可能造成 bag 元数据缺失。软件阻尼不是独立硬件急停，也不提供站立平衡。录制没有改变现有停止规则。

说明：`--record-raw-hz 10` 只降低原始 1280×720 图像的保存频率，不降低相机采集或策略输入频率。策略输入图约 30 Hz，推理／注意力约 50 Hz。默认原图未压缩，120 秒预计 3.5～4 GB；建议保留至少 6 GB 空间。`--record-max-mib 4096` 到达约 4 GiB 后只停止录制，控制程序继续。容量包含等待阶段，延长实验应相应调大上限并检查磁盘。

```bash
df -h /home/unitree/vibe_g1
cat "$VIBE_TRIAL/bag/recording_status.json"
```

说明：磁盘检查可在实验前运行，状态文件在录制中约每秒更新。正常结束应为 `complete: true`、`error: null`、`limit_reached: false`，`*_dropped` 与 `queued_not_written` 应全为 0。录制队列满时丢弃录制数据，容量上限／低磁盘／写盘错误结束录制，不触发电机指令；会留下计数或错误。USB 相机故障仍属于原有相机健康检查，录制不修复 USB 问题。

说明：如果特别需要每一张原图，把 `--record-raw-hz 10` 改为 `30`，同时增大容量上限。这会明显增加磁盘负载，120 秒约 10 GB；当前实时性能验收采用 10 Hz，不代表 30 Hz 已通过。热力图仍以每一步实际使用的策略图像为准。

## 2. G1：退出策略后生成回放网页

```bash
bash hardware/deployment/g1_python.sh \
  -m vibe_grail_joint.deploy.visualize_bag \
  --bag "$VIBE_TRIAL/bag" --output "$VIBE_TRIAL/preview" --fps 5
```

说明：在刚才的终端运行；如果换了终端，先将 `VIBE_TRIAL` 设为本次实验的实际绝对路径。生成后会输出 `index.html` 路径、缺图计数与注意力缺失计数。输出目录必须是新目录。导出只读 bag，不接管机器人；退出实时控制后再执行，避免占用推理资源。

```bash
bash hardware/deployment/g1_python.sh \
  -m vibe_grail_joint.deploy.visualize_bag \
  --bag "$VIBE_TRIAL/bag" --output "$VIBE_TRIAL/preview_detail" \
  --start 60 --seconds 8 --fps 50
```

说明：把 60 换成异常附近的秒数，生成该段逐策略步回放。时间零点是第一步推理请求，包含初始化等待，不是按 A 的时刻。`--fps` 只影响导出采样，不改变 bag；50 Hz 推理可能重复使用同一张约 30 Hz 图像。默认预览最多 1500 帧，长片段可分段或增加 `--max-frames`。

## 3. 电脑：拷回并打开回放

```bash
cd /home/wjw/vibe
TRIAL_NAME=trial_替换成本次实际时间
mkdir -p records/g1_recordings
scp -r "g1:/home/unitree/vibe_g1/records/$TRIAL_NAME/preview" \
  "records/g1_recordings/${TRIAL_NAME}_preview"
xdg-open "records/g1_recordings/${TRIAL_NAME}_preview/index.html"
```

说明：先把 `TRIAL_NAME` 换成 G1 实际生成的目录名，例如 `trial_20261028_223000`。上面只取网页和缩略图，传输量较小；网页可直接本地打开，无需服务器或联网。若按总命令文档生成了 `run_.../trial_...` 两层目录，请把 scp 源路径改成实际目录。

```bash
scp -r "g1:/home/unitree/vibe_g1/records/$TRIAL_NAME" records/g1_recordings/
```

说明：要保存完整实验原始数据，执行这一条，复制整个实验目录，包括 bag、日志和状态。不要只拷贝 `.db3`；保留 `metadata.yaml`。原始彩色像素保存在 bag，网页左侧 JPEG 只是缩小预览。

```bash
PYTHONPATH=hardware/src .venv/bin/python \
  -m vibe_grail_joint.deploy.visualize_bag \
  --bag "records/g1_recordings/$TRIAL_NAME/bag" \
  --output "records/g1_recordings/${TRIAL_NAME}_local_preview" --fps 5
```

说明：这是在电脑上重新生成回放的替代命令，适用于已经拷回完整 bag 的情况，不需要连接机器人。

## 4. 网页里怎么看

左侧是同一源帧的原始彩色图；中间是模型实际接收的 112×64 图；右侧是热力图叠加。滑块、上一帧／下一帧和播放按钮控制时间。原图默认 10 Hz，部分策略步骤没有对应 raw，会明确显示缺失，不用前后帧代替。策略图和特征按源图时间戳精确对应，注意力按策略步对应。

热力图下拉框可选择三组注意力均值、任务指令 `q_task_cmd`、本体状态 `q_proprio`、视觉 `q_cls`，或编码器特征 L2 幅值。默认显示蓝—青—黄—红的连续热区，透明度 0.55，默认不画网格；双线性插值只用于显示，原始数值和 4×7 分辨率不变。选择“原始色块”并打开 patch 网格，可核对原始位置，也可以打开白色图像梯度边缘。

默认色标使用“本帧范围”，数值范围显示在图下；跨帧比较注意力强度时可切换“注意力固定 0～1”。编码器 L2 幅值仍按该帧最小/最大值显示，不是概率。不能仅凭自适应色标下的颜色深浅判断权重绝对值变化。

本地已有 [真机/仿真对照页](../records/g1_sim_compare_20261029/index.html) 和详细回放已更新；旧页面保留为各目录的 `index_blocks_before_smooth_20261029.html`。新关键帧见 [平滑热区对照图](../records/g1_sim_compare_20261029/attention_comparison_smooth.jpg)，新视频见 [真机平滑注意力视频](../records/g1_sim_compare_20261029/real_policy_attention_smooth.mp4)。本次只更新本地显示，未连接或同步 G1。

注意力来自原策略同一次推理导出的 `attn`，不是根据图像边缘生成的示意图。当前模型只有 4×7 个 patch，空间模块已经混合信息，因此这张图只提供注意力落在哪些位置的线索，不能证明因果贡献或正确识别了台阶边缘。L2 只表示特征幅值；白色线只表示图像梯度，包含纹理、阴影等边缘。此工具不自动标注台阶，也不输出“上楼安全”的结论。

## 5. bag 内容与离线复算

| Topic | 类型／内容 |
|---|---|
| `/vibe/camera/color/image_raw` | `sensor_msgs/Image`，1280×720 BGR8 原始彩色像素，无深度 |
| `/vibe/camera/policy/image` | `sensor_msgs/Image`，真实裁剪／投影后的 112×64 BGR8 |
| `/vibe/camera/{color,policy}/camera_info` | 实际内参与策略投影内参；没有伪造相机安装外参或 TF |
| `/vibe/camera/frame`、`/vibe/vision/frame` | JSON，设备帧号与源图像时间 |
| `/vibe/vision/patches`、`/vibe/vision/cls` | 浮点数组，28×192 patch 特征与 CLS |
| `/vibe/policy/attention` | 浮点数组，3×28，查询顺序为 task_cmd、proprio、cls |
| `/vibe/policy/step` | JSON，策略步、源帧、图像时间、速度／转向、遥控器按键等 |
| `/vibe/policy/inputs/*` | 每个真实 ONNX 输入，包括历史与参考信息，可复算动作 |
| `/vibe/policy/actions`、`/vibe/policy/raw_target` | 原策略动作与 PD 目标，尚未经过可选的目标处理 |
| `/vibe/robot/joint_states`、`/vibe/robot/imu` | 真实关节角／速度、机身四元数／角速度；加速度不可用标志为 -1 |
| `/vibe/control/events` | JSON，初始化、启动、停止、控制循环；`control_tick.q_target` 是实际选用目标 |
| `/vibe/recording/metadata`、`/vibe/recording/status` | 模型哈希、参数、时间映射、录制计数／错误 |

说明：bag 时间由同一次启动时的单调时钟与系统时钟偏移转换；`policy/step.image_stamp_ns` 对应源图时间，不能直接用策略步时间替代。JointState／Imu 的 header 保留状态接收时间。没有记录相机深度，因为当前模型只使用 RGB；没有记录关节力矩，因为当前部署状态结构没有该测量。

```bash
bash hardware/deployment/g1_python.sh hardware/deployment/check_recorded_bag.py \
  --bag "$VIBE_TRIAL/bag" --bundle models/vggt_hierarchical_rgb64 \
  --samples 12 --output "$VIBE_TRIAL/replay_check.json"
```

说明：G1 退出控制后运行，或在电脑用 `PYTHONPATH=hardware/src .venv/bin/python` 替换前面的解释器入口。只读取 bag 与模型，使用 CPU 重算采样步骤，检查输入特征是否精确对应源图记录、动作与编码器误差是否在 1e-5 内。CPU 与 GPU 的注意力微小差异单独报告，不作为动作正确性的指标。缺记录或模型哈希不一致会报错，不能把这种报告当作通过。

## 6. 只录制、绝不接管电机的测试入口

```bash
cd /home/unitree/vibe_g1
VIBE_TRIAL="$PWD/records/readonly_bag_$(date +%Y%m%d_%H%M%S)"
taskset -c 2,3 bash hardware/deployment/g1_python.sh hardware/deployment/readonly_writer_probe.py \
  --bundle models/vggt_hierarchical_rgb64 \
  --engine-dir engines/vggt_hierarchical_rgb64_trt10_fp32 \
  --seconds 30 --hold-seconds 5 --policy --loopback-dds \
  --inference-cpus 4,5,6,7 \
  --record-bag "$VIBE_TRIAL/bag" --record-raw-hz 10 \
  --output "$VIBE_TRIAL/readonly.json"
```

说明：与运动程序互斥使用。该入口读取真实相机与关节并计算策略，禁止 SDK 发布者；500 Hz 模拟报文只写入本机 lo、独立 DDS 域 197 和丢弃主题。它不会初始化、切换机器人模式或驱动电机。用第 2 节命令可回放此记录，但静止机器人上的动作输出不能代表实际运动效果。
