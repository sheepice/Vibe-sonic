# 轻量 G1 真机部署入口

基于 `wjw:/mnt/workspace/wjw/GRAIL/vibe_grail_joint` 的部署源码快照，用于当前 `models/velocity3999` 包。已部署独立 Python 3.11、V4L2 相机绑定、TensorRT GPU 推理及遥控器输入。来源与本地修改哈希见 [provenance.json](provenance.json)。

G1 日常启动使用 `bash hardware/deployment/g1_python.sh -m vibe_grail_joint.deploy.run ...`，为本进程选择系统 TensorRT 10.3 / CUDA 12.6，避免交互终端加载家目录中的不兼容库。

已修复 DDS 默认 CDR 版本参数 `None` 绕过快速打包的问题；默认版本与两个显式版本均对照官方 SDK 校验，线上字节格式保持一致。发送循环保留原 500 Hz 和 20 ms 截止时间，退出日志新增 `writer` 分阶段耗时。`deployment/readonly_writer_probe.py --loopback-dds` 使用独立域 197、仅 `lo` 回环网卡和丢弃主题检查实际 DDS 写出，禁止 SDK 发布器，不发送机器人电机指令。

日常操作见 [运行命令](../docs/G1_RUN_COMMANDS_ZH.md)，详细步骤见 [真机教程](../docs/G1_REAL_ROBOT_GUIDE_ZH.md)，最新性能与验证边界见 [优化状态](../docs/G1_OPTIMIZATION_STATUS_ZH.md)。

- `deploy.check`：离线模型和计算时间检查，不接机器人。
- `deploy.camera`：本机 USB D435i RGB 检查。
- `deploy.run --mode inspect`：只读 LowState，不创建 LowCmd publisher。
- `deploy.run --mode shadow`：真实传感器影子推理，不发布电机命令。
- `deploy.run --mode control`：需 `--execute`，以及实测 `--calibration` 或显式 `--allow-unmeasured-camera`，再通过初始化按键接管。跳过标定仍记录为未标定。初始化从接管后的真实关节角缓慢过渡到名义姿态并保持（至少 3 秒），不自动启动策略。默认传递原始 PD 目标，无需旧的受保护验收报告；显式加 `--actuator-guards` 可恢复目标保护，此时须提供匹配的 `--acceptance-report`。

推荐 `--input remote`：全部松键后 R1＋START 长按 1 秒初始化；姿态就绪后全部松键，单独长按 A 1 秒启动策略；松开后方向键前后／左右转，B 正常停步但继续策略。初始化姿态保持没有主动平衡，必须维持支撑；策略启动前双脚落地。键盘备选为 Enter 初始化、P 启动策略、W 前进、Space 停步。

X 键、窗口红色按钮或遥控 SELECT 锁存停止。已接管时停止策略推理，写线程持续尝试阻尼，支撑后用 Esc 或遥控 Y 长按 2 秒退出；松键或初始化/策略键不会恢复控制。X 只在 pygame 控制窗口内有效。通信、无效数据、电机健康检查继续有效；软件阻尼依赖程序/通信正常，不保持站姿，也不是独立硬件急停。遥控器无线断连行为尚未验证。

本目录不是完整训练工程；`simcheck`、重新导出和训练仍需完整环境。键盘 S/Q/E 扩展仍只属于本地仿真。助手已完成 G1 真实传感器和五分钟隔离 DDS 回环测试，没有启动真机电机控制；用户已执行初始化，策略平衡、现场停止响应及楼梯仍未验证。

运行本地全部测试：`PYTHONPATH=hardware/src:. .venv/bin/python -m pytest hardware/tests -q`（项目根目录）。SDK 控制测试全部使用假传输，不接真实电机；包括初始化后等待、拒绝提前启动、A 单独启动、方向键及 SELECT 锁存。
