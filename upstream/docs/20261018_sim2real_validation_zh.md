# 2026-10-18 sim2real 代码、模型与验证记录

项目绝对路径：`/mnt/workspace/wjw/GRAIL/vibe_grail_joint`。

**部署代码和教程已补齐，尚未完成真机验证。当前速度模型从静止起步时触发部署保护；作者原版加在线参考能在本次平地试验中起停，但三个楼梯试验均未通关。** 这两点不能被此前行走中途初始化的成功录像替代。

操作教程：[当前 SONIC＋ViBe 框架](20261018_sim2real_g1_d435i_zh.md)、[作者原版 ViBe](20261018_original_vibe_real_robot_zh.md)。本文件保存结果与限制，具体命令以这两份教程为准。

## 1. 代码和权重来源

新增 `src/vibe_grail_joint/deploy/`、`deployment/`、三份 `tests/test_deploy_*.py` 和本文相关教程。没有修改原有训练、评估、SONIC 或作者 C++ 源文件，没有终止既有训练。官方 C++ 改动仅作为独立 patch 保存；真正应用时，脚本必须先核对源文件 hash 并备份。

| 包 | 位置（相对项目根目录） | policy SHA-256 |
|---|---|---|
| 本地速度模型 | `exports/20261018_sim2real_velocity3999` | `27da79327d46cc3aa1cf1fbe685f1d52071223b15737fc82aa5412366401bae4` |
| 作者原版＋本地在线参考接口 | `exports/20261018_sim2real_official_vibe` | `f1a36fca7adc887f7777a0340dbb62905d4ef6e47617b3fe95e82d76b686f4d9` |
| 作者发布文件原样缓存 | `exports/20261018_vibe_official_walk` | 与上一行相同 |

作者文件来自 [lkrajan/vibe](https://huggingface.co/lkrajan/vibe)，由 [官方工作区](https://github.com/drcl-deploy/vibe_ws) 的锁定清单定位到 revision `ff79137d921d0124fea5877e01d1065f10e852a3`。policy、policy manifest、Theia、Theia manifest 共四个文件均核对大小和 SHA；receipt 保存在官方缓存目录。

两包的 Theia ONNX 文件分别保存自己的 hash，不声称字节相同。共同 planner SHA 是 `39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea`。运行时核对本包全部文件，不能随手互换 encoder。

本地速度 checkpoint 为：

```text
/mnt/workspace/wjw/GRAIL/vibe_grail_joint/runs/20261018_vibe_posture_speed_refine/train_03500_04000_attempt0/model_3999.pt
SHA256 8e6e64658f45ce6971cdfc832a77ac43fad3796de81a2103b18e1990219c671f
```

它的完整 SONIC、ViBe 适配器和新增速度分支均导入 ONNX，没有忽略速度编码器来强行沿用旧导出入口。

## 2. 自动测试与导出数值

`records/20261018_sim2real/tests_final.log`：**37 passed**。覆盖真实容易出错的接口边界：

- 29 关节顺序、历史项布局、未来参考特殊 reshape、参考自身坐标系 twist、相对朝向和速度指令。
- 键盘松开回零、失焦/停止锁存；目标限位、变化率和非有限值。
- 模拟 SDK 消息下：只读不创建 LowCmd publisher、CRC 错误/重复 tick 不刷新状态、PR 模式/型号/29 电机字段/剩余槽位、命令 CRC 填写顺序、状态和输出超时、遥控停止、型号变化、写入失败后阻尼。
- 仿真证据必须匹配模型文件和 PD/相机/保护参数；不接受跌倒、提前退出、未停车、仍漂移、合成状态代替物理仿真、超出已测速度。
- RGB 投影基本数值和官方 TCP 包格式/颜色通道无损往返。

SDK 测试使用 fake message 和 fake publisher，**未打开 DDS 或发送电机指令**。它们证明软件分支行为，不证明真实 SDK 二进制、固件和网络已通过。

`records/20261018_sim2real/parity_fp32.json` 是真实物理环境中的 100 步输入对齐测试，测得最大绝对误差：

| 对照 | 最大误差 | 解释 |
|---|---:|---|
| NumPy 非视觉观测 vs 训练观测 | 3.73e-7 | 历史、参考、速度等布局通过 |
| 同输入 CPU actor vs ONNX | 1.91e-6 | 保留速度分支后的导出通过 |
| CPU FP32 Theia vs ONNX Theia | 1.17e-5 | 编码器导出通过 |
| 训练 FP16 patch vs 部署 FP32 patch | 2.54e-2 | 精度差异，单独保留 |
| 训练 FP16 CLS vs 部署 FP32 CLS | 4.13e-3 | 精度差异，单独保留 |

最初 `parity.json` 将训练 FP16 与 ONNX FP32 特征直接按严格同精度阈值比较，失败记录仍保留。随后拆开同精度导出检查和训练/部署精度差异，没有删除真实差异。导出阶段另有 32 组随机输入检查，结果保存在速度包 manifest 中。FSQ 有离散边界，小数值误差不保证长时闭环轨迹完全一致。

新 Python 文件已通过 `compileall`。官方 C++ patch 通过固定 pin 的 `git apply --check`，日志为 `records/20261018_sim2real/official_patch_check.log`。**本机没有 ROS 2，因此没有编译该补丁，也没有跑官方 ROS tracker。**

收尾核查 `artifact_cli_audit.json` 确认两个模型包的全部文件 hash 和 ONNX checker 通过，四个作者原始文件仍与下载锁一致，9 个命令行入口的 `--help` 可运行。运行时模块的导入没有加载 PyTorch、mjlab、MuJoCo、SDK、RealSense 或 pygame；硬件依赖只在实际使用时加载。

## 3. 静止起步、W、松键停步的物理测试

场景从默认静止站姿开始；真实 ONNX actor 和 RGB encoder 驱动物理仿真。W 指令在第 2 秒开始，目标速度 0.25 m/s，每步最多增减 0.01 m/s；第 8 秒释放，计划运行 12 秒。目标经过与新增部署入口相同的限位投影、slew 和跟踪误差检查，实际应用的动作回写历史。没有实际 DDS、相机采集延迟或真实电机。

| 模型/记录 | 结果 | 位移/保护 |
|---|---|---|
| 原版 `official_flat_acceptance_v2.json` | 600 步、12 s，无跌倒、无保护中断，完成前进和停车 | 前进 1.465 m；最后 1 s 水平位移约 0.004 m |
| 速度版 `velocity_flat_acceptance_v2.json` | 148 步，2.96 s 保护中断，未完成起停 | 前进约 0.137 m；左踝 pitch 目标误差 1.208 rad，超过 1.2 rad |

这两个报告都含当前模型/执行配置签名。`validate_sim_acceptance` 对原版 0.25 m/s 放行，对速度版拒绝；原版报告也不能给速度版或更高速度放行。原版平地最大目标限位投影约 0.241 rad，已经接近 0.25 rad 阈值，因此也不应把它描述成有充足硬件裕量。

原始 `velocity_free.json` 在不加部署目标保护时运行了 12 s、未跌倒，但有 61 次目标越界检查命中。它不是可用于启用电机的通过报告。较早的 `keyboard_flat_guarded.json` 在加 slew 前报目标突变；加 slew 后仍存在上述左踝跟踪误差，不能声称已解决起步问题。

RGB 小视频是模型输入视角的诊断记录，路径在同名 JSON 旁，后缀 `.rgb.mp4`。为了视频编码把 63 行补到 64 行，补的这一行只用于视频，**不进入策略输入**。它们不是之前固定外部视角的展示视频。

两项 v2 平地和三项楼梯共 5 段小视频已逐帧解码核查，记录为 `diagnostic_video_audit.json`。视频以 10 FPS 取样并向整帧取整，显示时长可能比 JSON 保护中断时刻多不到 0.1 s；计时和成败以 JSON 为准。

## 4. 原版＋在线参考的楼梯尝试

统一请求 `39 s`，W 行走时长 `35 s`，速度 0.25 m/s。保留三次全部结果，成功 **0/3**；这是本项目新增在线参考链路的测试，不是作者固定参考路径的论文复现。

| 场景 | 每级高度 / 踏面 / 级数 | 结束时间 | 结果 |
|---|---|---:|---|
| 上楼 | 10.10 cm / 38.54 cm / 6 | 8.90 s | 第一阶前附近，目标投影需 0.2900 rad，超过 0.25，保护结束 |
| 下楼 | 9.81 cm / 29.09 cm / 6 | 9.12 s | 第一阶附近，投影需 0.3793 rad，保护结束 |
| 上后下 | 9.88 cm / 33.83 cm / 每侧 5 | 6.98 s | 第一阶前附近，投影需 0.2836 rad，保护结束 |

报告分别是 `official_up.json`、`official_down.json`、`official_up_down.json`，位于 `records/20261018_sim2real/`。没有在中断之后继续推演，因此“未发生跌倒”只表示中断前未倒，不能推断放宽保护后也不会倒。程序退出码 0 只表示完成报告保存。

这些报告是签名字段加入前的早期记录，仅用于结果分析，不能用于当前控制入口的自动前置检查。当前控制放行依据只有完整的平地 v2 报告；**没有楼梯通关报告**。

## 5. 为什么旧录像可以走，这里又失败

旧任务的在线参考确实由 SONIC planner 持续生成，不是楼梯动作真值。不过训练 `velocity/planner.py` 初始化先生成行走 0.8 s，`velocity/command.py` 把在线环境直接重置到对应参考的姿态、关节速度和根速度，再叠加不同相位/路面。因此行走中途的楼梯成功，不等于静止站立→迈第一步→楼梯→停车这一整段成功。

新增部署参考从静止姿态开始，不跳过前 0.8 s；指令变化驱动同一滚动参考。部署另加了目标保护，训练原来的 PD target 没有相同的限位投影和 slew。这两处都改变闭环条件，是本次失败的具体排查方向。不能由现有结果断言唯一原因是某个 reward，也不能断言 frozen SONIC 一定不能迈楼梯。

下一轮训练应把这些条件纳入：从默认站姿初始化、站立/起步/行走/停车切换、短暂停顿后重启；在物理训练中使用实际部署的目标投影/变化率和动作历史；加入相机 30 FPS、延迟、帧复用及实测外参。继续保持冻结 SONIC 和上身分组跟踪约束，训练视觉适配器/速度分支，使模型学会在实际执行约束下调整动作。优先解决平地起停，再推进小台阶，不靠提高保护阈值取得表面成功。

## 6. 性能结果的含义

全部数值来自这台训练服务器的 CPU ONNX，不是 G1 机载平台。离线固定输入检查每次 500 帧：

| 记录 | 观测＋图像编码＋actor P99 | planner P99 |
|---|---:|---:|
| 速度版原检查 `records/20261018_sim2real_offline_check.json` | 5.97 ms | 59.3 ms |
| 原版与其他任务并发时 `official_offline.json` | 6.25 ms | 443.6 ms |
| 原版空闲复测 `official_offline_idle.json` | 6.69 ms | 69.1 ms |

并发时 planner 已超过 200 ms 周期；不能拿空载数字覆盖这个失败。程序把 planner 放工作线程、检查参考年龄；输出过期时停止，不继续沿用旧参考。整个控制进程也需要独立 CPU 资源和实际负载测试。

新加 `check --wall-clock` 后，初次未预热的 600 帧记录 `official_wall_clock.json` 的 actor 整体路径 P99 7.86 ms，但最大 80.49 ms，发生 3 个超过一周期的迟到 tick。因此增加显式 actor/encoder 预热，放在键盘接管前执行。一次预热后的测试仍遇到 planner 过期，日志 `records/20261018_sim2real_official_wall_clock_warm.log` 保留；预热不等于解决所有系统调度问题。

后续 `official_wall_clock_warm_diagnostic.json` 完成了 600 帧：actor 路径 P99 8.01 ms，最大 41.96 ms，仍有 1 个超过一周期的迟到 tick；planner P99 180.63 ms、最大 186.51 ms。这一次没有 planner 过期，但 41.96 ms 已超过控制入口的 40 ms 停机阈值，不能记为实时验收通过。检查脚本现在也会在失败时保存已经测到的耗时和 planner 状态。

**目前没有完成包含真实相机、DDS、日志、控制 writer 的 50/500 Hz 时序验收。** 后续应使用最终部署计算机、安排专用 CPU 资源并记录最大延迟；若 Python 路径仍不能稳定满足周期，应将相同观察契约和速度分支接到独立进程/原生控制循环再验。不能通过提高 watchdog 阈值掩盖调度卡顿，也不能据一次合成检查宣称硬实时保证。

## 7. 原八卡训练状态及已有成绩

`runs/20261018_vibe_posture_speed_refine/stage_04000_feedback.json` 记录：8 个 rank 均 `completed`、完成更新数 4000、`frozen_unchanged=true`。总环境数 8×2040＝16320。uniform 和 irregular 的正常视觉评估均为楼梯 180/180，加 12/12 平地。

监督状态 `completed_with_evaluation_failures` 来自评估：关视觉 token 的一例 irregular 跌倒，以及困难原始 GRAIL 速度任务失败，不是训练在第 77 轮异常退出。上身修正的分析、研究依据和 3000 轮前后量化见 [上身修正评估](20261018_posture_3000_evaluation_zh.md)；之后的 1000 轮在保持相同姿态约束的同时加强速度跟踪。

这些结果支持“已有行走状态下的训练任务改善并完成八卡训练”，不支持“新增真机部署条件已经全部通过”。本次没有为修正部署失败偷偷重新开始训练或更改原训练配置。

## 8. 尚待实机/目标计算机完成的项目

1. 安装对应 Unitree SDK、RealSense SDK、ROS 2 和官方 C++ 工作区；编译可选补丁并完成官方样例仿真。
2. 测量 D435i RGB 光心相对 torso 外参，核对 FOV/畸变/光照/遮挡，确认图像数据年龄。
3. 实际 LowState 的关节顺序、IMU、CRC、型号、遥控按键，以及有线 DDS 的读取/控制权交接。
4. 含实际相机、日志和 SDK writer 的时序/断连测试；验证机器人侧独立停机行为。Python 进程自身死亡时无法依靠自身 watchdog。
5. 支撑下验证站姿过渡、PD/扭矩限值和电机响应；平地起停、低台阶上/下逐项通过后再扩大范围。

优先试原版时，先走官方固定参考基线，选与参考相符的几何。希望只按 W 自主适应楼梯时，使用新增在线参考路线，并先补齐本文件记录的起停/限位/时序问题；不能直接把原版固定参考的成功预期套到 W 路线上。
