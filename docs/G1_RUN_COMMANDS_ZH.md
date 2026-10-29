# G1 真机运行命令（RGB64 hierarchical model_2400，2026-10-28）

## 1 电脑端登录 G1

```bash
ssh g1
```

说明：登录机器人电脑。模型、相机和控制程序都在 G1 上运行。电脑上的 `./run.sh` 是仿真入口，真机实验不用启动它。

说明：推荐使用官方遥控器输入。下文 `--input remote` 可直接通过 SSH 启动，不需要图形桌面；按键来自 G1 的 LowState。

## 2 G1 端每个新终端先执行

```bash
cd /home/unitree/vibe_g1
export G1_IFACE=enP8p1s0
export D435_SERIAL=254522074774
export VIBE_ENGINES="$PWD/engines/vggt_hierarchical_rgb64_trt10_fp32"
export VIBE_LOGDIR="$PWD/records/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$VIBE_LOGDIR"
```

说明：进入已部署的项目，设置本机网口、D435i 序列号及本次日志目录。后续均通过 `bash hardware/deployment/g1_python.sh` 调用本项目的 Python 3.11，无需启动训练程序。该脚本只为当前进程固定系统 TensorRT 10.3 / CUDA 12.6 库路径，避免交互终端中的家目录 TensorRT 10.7 与 `libcudla` 冲突；不改系统驱动和 `.bashrc`。

```bash
sudo /usr/bin/jetson_clocks --store "$VIBE_LOGDIR/clocks_before.conf"
sudo /usr/bin/jetson_clocks
```

说明：在这台 Orin NX 当前电源模式内固定 CPU/GPU/内存频率，这是已通过只读压力测试的运行条件之一，不会启动电机或修改电机控制器。每次实验只保存一次原配置；结束并支撑、退出所有程序后按第 7 节恢复。不需要修改系统驱动或重启。

## 3 G1 端检查相机和机器人状态

```bash
bash hardware/deployment/g1_python.sh -m vibe_grail_joint.deploy.camera \
  --bundle models/vggt_hierarchical_rgb64 --serial "$D435_SERIAL" --seconds 10 \
  --output "$VIBE_LOGDIR/camera"
```

说明：采集相机并保存报告和 `policy_rgb.png`，不会驱动机器人。检查结束后再启动下一条命令，避免两个程序同时打开相机。

说明：若再次出现 `stale_camera`，保留本次日志与 bag，先停止该次实验。2026-10-28 23:31 的记录证实初始化接管前发生 USB 链路错误和原图断帧；已为这台 D435i 持久化禁用自动休眠，但尚未验证恢复。不要将一次短暂相机检查的约 30 FPS 当作稳定运行证明，也不要跳过图像新鲜度检查继续运动；需排查并更换可靠 USB 3 数据线／接口后先做只读复测。详见 `G1_OPTIMIZATION_STATUS_ZH.md`。

```bash
bash hardware/deployment/g1_python.sh -m vibe_grail_joint.deploy.run \
  --bundle models/vggt_hierarchical_rgb64 --interface "$G1_IFACE" \
  --mode inspect --seconds 20 \
  --log "$VIBE_LOGDIR/inspect.jsonl"
```

说明：只读取关节、IMU 和遥控器状态，不发送电机指令。核对关节数量为 29、`crc_failures=0`、状态持续更新。逐个按键时用 `remote_button_names` 和 `remote_buttons_hex` 核对 R1、START、UP、DOWN、LEFT、RIGHT、A、B、SELECT、Y；先确认厂商当前模式允许该按键操作。机器人静止并有支撑、现场允许后，按住官方遥控器 SELECT 约 2 秒再松开：`remote_select` 应先变为 `true` 再回到 `false`，而 `stop_latched` 应保持 `true`，`stop_reason` 为 `remote_emergency_stop`。这一步核对映射和程序锁存逻辑，不证明带电机控制时的实际停止效果；当前尚未进行现场按键验证。

## 4 G1 端只读复测与遥控器影子模式

```bash
taskset -c 2,3 bash hardware/deployment/g1_python.sh hardware/deployment/readonly_writer_probe.py \
  --bundle models/vggt_hierarchical_rgb64 --engine-dir "$VIBE_ENGINES" \
  --seconds 300 --hold-seconds 30 --policy --loopback-dds \
  --inference-cpus 4,5,6,7 \
  --output "$VIBE_LOGDIR/readonly_writer.json"
```

说明：可在 SSH 运行；脚本对应本机 `vggt_hierarchical_rgb64`、`enP8p1s0` 与 D435i `254522074774`。先模拟 30 秒姿态保持，再运行站停、前后和左右转向的模型计算，所有动作输出均丢弃。500 Hz 报文只写到 `lo` 回环网卡上的独立 DDS 域 197、主题 `vibe_diagnostic_discard_only`，禁止 SDK 发布器及电机接管。检查 `software_health_pass=true`、`first_fault=null`、`failure=null`、`crc_failures=0`、`sdk_publisher_attempts=[]` 和 `motor_transport=false`。它覆盖实际 DDS API 调用，但没有验证到电机的网络链路、平衡或真实停止效果。结束后再打开影子模式，避免争用相机。

```bash
taskset -c 2,3 bash hardware/deployment/g1_python.sh -m vibe_grail_joint.deploy.run \
  --bundle models/vggt_hierarchical_rgb64 --interface "$G1_IFACE" \
  --mode shadow --input remote --inference-cpus 4,5,6,7 --provider tensorrt --engine-dir "$VIBE_ENGINES" \
  --serial "$D435_SERIAL" \
  --speed 0.4 --turn-rate 0.45 --seconds 60 \
  --log "$VIBE_LOGDIR/shadow.jsonl"
```

说明：读取真实相机和关节，不发送电机指令。先释放全部按键，长按 R1＋START 至少 1 秒进行虚拟初始化，全部松开，再单独长按 A 至少 1 秒启动影子策略，松开后测试方向键、B、SELECT。策略日志应显示 `lowcmd_enabled=false`。不按 A 就不会进入策略控制循环。SELECT 会结束影子测试，重启才能继续。实体按键、无线断连和电机停止响应尚未验证。

## 5 由你手动启动受支撑运动

说明：首次采用官方低层部署流程：先让吊具承托机器人，在零力矩状态按 **L2＋R2** 进入调试模式（此时关节阻尼），再松开所有按键，启动下面的程序。切换时不能让机器人无支撑站立。宇树模式键与本程序的 R1＋START 初始化、A 启动策略是不同阶段。[宇树官方实物部署说明](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.zh.md)

说明：按用户要求，本节使用显式 `--allow-unmeasured-camera`，暂时跳过相机安装外参检查，不需要创建标定文件。启动会显示 `UNMEASURED CAMERA`，日志记录未标定；相机仍需正常工作。未标定可能导致视觉判断和动作错误，首次仅在双脚落地、吊具支撑的平地试验，不直接上楼梯。

说明：本次使用与当前仿真相同的 RGB64 hierarchical `model_2400`：112×64 彩色输入、28×192 视觉特征、3 层空间注意力；不需要部署 VGGT 教师。必须使用上面对应的新引擎目录。保留此前已修好的 DDS 序列化及主进程核心 2/3、推理核心 4–7 分配。新版本的 20 ms 写线程迟到和 40 ms 主循环超时只记告警，实际通信、数据和电机故障仍停止；最新验证范围见 `docs/G1_OPTIMIZATION_STATUS_ZH.md`。程序完全退出后重新运行命令即可加载新代码，不用重启机器人。

```bash
STAIR_SPEED=0.4
VIBE_TRIAL="$VIBE_LOGDIR/trial_$(date +%Y%m%d_%H%M%S)"
taskset -c 2,3 bash hardware/deployment/g1_python.sh -m vibe_grail_joint.deploy.run \
  --bundle models/vggt_hierarchical_rgb64 --interface "$G1_IFACE" \
  --mode control --execute --release-sport-mode --input remote --inference-cpus 4,5,6,7 --provider tensorrt \
  --engine-dir "$VIBE_ENGINES" \
  --serial "$D435_SERIAL" \
  --allow-unmeasured-camera \
  --speed "$STAIR_SPEED" --turn-rate 0.45 --backward-speed 0.1 --seconds 120 \
  --log "$VIBE_TRIAL/control.jsonl" \
  --record-bag "$VIBE_TRIAL/bag" --record-raw-hz 10 --record-max-mib 4096
```

说明：本命令同时录制原始彩色图像、策略实际输入、视觉特征、注意力、关节与控制事件。后台录制进程使用核心 0/1；原图默认 10 Hz，120 秒约 3.5～4 GB。每次完整执行上述代码块会生成新的 `VIBE_TRIAL`；同一 bag 目录不能重复使用。正常支撑、SELECT 停止并长按 Y 退出后，再导出回放；不需要启动 ROS。录制达到容量上限或写盘失败只停止录制，不改变控制状态，终端会提示。回放、复制与各 topic 说明见 [G1_RECORDING_ZH.md](G1_RECORDING_ZH.md)。删除 `--record-bag`、`--record-raw-hz` 和 `--record-max-mib` 即可关闭录制。

说明：以后完成实测标定时，将 `--allow-unmeasured-camera` 替换成 `--calibration "$PWD/calibration/my_g1_camera.json"`，两个参数不能同时使用。

说明：若前一次启动已报错退出，重新执行整个控制代码块，包括重新设置 `VIBE_TRIAL`，保证日志与 bag 都使用新目录；已有实验不会覆盖。

说明：仅在前述条件满足后使用。启动后释放全部按键，长按 R1＋START 至少 1 秒接管并松开，等待终端显示的姿态过渡时长结束（至少 3 秒，按实际角度差自动延长）、终端提示 `Nominal pose ready: PD HOLD ONLY`。此时保持关节姿态，**没有运行平衡策略，也不会自动启动策略**。保持吊具支撑，确认双脚接触平整地面、接近名义站姿且静止；全部松键，再单独长按 A 至少 1 秒，看到 `Policy started at zero velocity` 后松开。先观察零速度站立，确认后才按方向键。初始化期间按住 A 不会预约启动，必须等姿态就绪后松开再按。

说明：进入调试模式时先用吊具承托。初始化期间保持支撑，不再要求起始关节接近名义站姿，也不要求手动掰到规定角度；如初始化时双脚悬空，姿态过渡完成后再缓慢下放至双脚接触平地，然后才能按 A。初始化仅是关节姿态过渡，不能依靠它无支撑站立；不要完全吊空后启动行走策略。程序不支持从坐姿自动站起，也不能自动检测脚底是否落地或吊具是否到位。120 秒包含初始化、等待 A 和策略运行；到时会退出，应提前正常停止并支撑机器人。

说明：初始化从接管完成后重新读取的 29 个关节角开始，用平滑插值过渡到默认姿态，目标角速度峰值不超过 0.25 rad/s。角度差越大，过渡越久；旧的关节角度、速度门槛已改成日志诊断，启动 A 时也不再因这些阈值拒绝。仍要求状态数据有效且保持物理支撑。这不是坐姿、躺姿的自动起身动作规划，不能保证关节路径不碰撞；初始化和 PD 保持都不主动平衡。

说明：此命令默认关闭额外关节目标限幅、变化率和跟踪误差保护，保留通信、数据与状态检查。本节命令已加入 `--release-sport-mode`，用于现场出现的 `Another robot mode is active`：只有你长按 R1＋START 触发初始化后，程序才请求退出仍活动的宇树高层控制器，并等待确认释放，再接管电机。切换前必须让吊具承托机器人，退出原控制器时会撤掉其主动支撑；释放失败仍会退出，不强行抢占。

```bash
STAIR_SPEED=0.25
# 或在对应速度已完成前序验证后：
STAIR_SPEED=0.35
```

说明：正常停止、支撑并退出后，修改速度，再完整运行上一条控制命令。单位为 m/s；0.15、0.25、0.35 是启动参数示例，不代表这台真机已验证通过这些速度。真机没有 `--terrain up` 参数，前方实际地形由相机进入模型。遥控模式已加入后退与转向，本次前进 0.40 m/s、后退 0.10 m/s、转向 0.45 rad/s；用 `--backward-speed`、`--turn-rate` 修改。这组新策略及速度尚未进行真机运动验证；此前旧策略的真机成功结果不等于新策略已验证。键盘 S/Q/E 仍只用于仿真。

## 6 运动与停止操作

| 遥控器操作 | 本程序效果 |
|---|---|
| 程序接管前，在吊具承托、零力矩状态按 **L2＋R2** | 宇树调试模式，关节阻尼；进入后全部松键 |
| 松开全部按键，再长按 **R1＋START ≥1 秒**，随后松开 | 初始化／请求接管；不是坐姿自动站起；等待按角度差决定时长的缓慢姿态过渡 |
| 姿态过渡完成、全部松键后，单独长按 **A ≥1 秒**，随后松开 | 启动策略，初始速度为零；启动前双脚落地、吊具支撑到位 |
| 策略启动后按住方向键 **上／下** | 前进／后退；前进用 `--speed`，后退默认 0.10 m/s |
| 按住方向键 **左／右** | 左转／右转，0.45 rad/s（约 25.8°/s）；不是横向平移 |
| 松开方向键 | 请求行走与转向速度逐步归零，仍继续策略站立 |
| **B** | 正常停步，继续策略站立；须松开全部方向键后才可再次行走；不退出策略、不切回姿态保持 |
| **SELECT** | 锁存软件阻尼停止；松键或重新初始化都不恢复 |
| 已停止锁存并支撑好机器人后，长按 **Y ≥2 秒** | 退出程序；之后不再持续发送阻尼 |

说明：SELECT 在 DDS 回调直接锁存，不依赖电脑窗口；它依赖本程序和通信，不是独立硬件急停。阻尼会失去站立支撑，不是冻结姿势。首次先用 inspect 核对按钮和固件原生停止方式。实体按键、无线断连清零和真实停止响应尚未验证；DDS 状态持续更新不能证明遥控器仍连接。

说明：异常时使用已验证的停止入口并让支撑承托机器人。不要关闭 SSH、断网或 `kill -9` 来急停。键盘备选为 `--input keyboard`，须在 G1 图形桌面运行：Enter 初始化，姿态就绪后 P 启动策略，W 前进，Space 停步，X 锁存阻尼，Esc 退出。

说明：新版不再因 `writer_deadline`、姿态偏差、倾斜或默认关闭的跟踪误差保护锁定阻尼；写线程迟到计入 `writer.deadline_overruns` / `first_deadline_overrun`，主循环超过 40 ms 记为 `loop_deadline_warning`。仍保留 LowState 超过 50 ms、控制目标超过 60 ms、相机或输入轮询超过 150 ms 未更新，以及无效数据、电机错误/过温、控制模式变化、写出失败和程序异常的停止。不能解释为“只要不按 SELECT 就绝不会停止”。故障锁存后 A 不恢复，先支撑，再长按 Y 退出，保留日志。120 秒到期和 Ctrl+C 仍会结束程序。

## 7 G1 恢复频率，电脑端取回日志

```bash
sudo /usr/bin/jetson_clocks --restore "$VIBE_LOGDIR/clocks_before.conf"
```

说明：仅在已经正常停步、支撑并退出控制／测试程序后，在 G1 执行，恢复本次实验前频率配置。不要在正在接管电机时恢复调频。


```bash
mkdir -p /home/wjw/vibe/records/g1_runs
scp -r g1:/home/unitree/vibe_g1/records/run_实际时间 \
  /home/wjw/vibe/records/g1_runs/
```

说明：将 `run_实际时间` 换成本次 G1 上生成的目录名。停止并退出测试后执行，不覆盖已有实验记录。
