# Vibe-sonic

Unitree G1 人形机器人视觉运动策略的 **sim2sim 验证环境**：把 GRAIL/ViBe 训练、VGGT 分层蒸馏得到的 RGB 策略与 SONIC 在线规划器导出为 ONNX，在原生 MuJoCo 动力学里完整闭环运行——不依赖 PyTorch、Isaac Lab，也不需要训练用的 VGGT 教师网络。

> 仓库不含模型权重与运行记录。克隆后向作者获取权重放入 `models/` 即可运行，详见下方[获取权重](#获取权重)。

- **当前默认模型**：`models/vggt_hierarchical_rgb64/`（checkpoint `model_2400.pt`，训练 2400 轮，2026-10-28 从训练服务器导出）。112×64 头部 RGB、4×7 共 28 个视觉 token、三层空间特征模块，策略 ONNX 中完整保留空间模块。
- **观测与控制**：29 关节 G1（12 腿 + 3 腰 + 14 臂）、10 帧历史、8 维速度/朝向指令，与训练配置一致；无深度输入。
- **在线参考**：SONIC 规划器 ONNX 以 5 Hz 在线生成 10 帧未来参考（锚点 pelvis，四元数 wxyz）。
- **物理**：MuJoCo `implicitfast`，步长 0.005 s；策略 50 Hz。执行训练所用原生 PD 目标与力矩限制，无根节点回写、无托举、无辅助外力，也没有额外部署保护。
- **验收**：自动判定无跌倒、脚部真实接触台阶、双脚越过终点、终点后停步 4 秒且漂移 < 5 cm。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `sim2sim/` | 核心：MuJoCo 仿真、ONNX 推理、观测构建、在线参考、网页服务、验收逻辑（`vendor/` 为严格校验的 ONNX 运行时，无 torch 依赖） |
| `models/` | 模型 bundle。JSON 清单在仓库内；`.onnx`/`.pt` 权重不入库，需另行放入 |
| `assets/` | mjlab G1 的 MuJoCo XML 与网格（约 19 MiB）、楼梯高度场样例 |
| `web/` | 本地控制页面，无 Node.js、无构建、无 CDN 依赖 |
| `hardware/` | 真机部署入口（独立 Python 3.11、V4L2 相机、TensorRT），含测试 |
| `scripts/` | 权重导出、浏览器自动测试、分析绘图等辅助脚本 |
| `docs/` | 真机教程、运行命令、录制与优化状态文档 |
| `upstream/` | 上游部署文档与接口代码快照、来源哈希与许可 |

## 环境准备

需要 Linux、Python 3.12（由 [uv](https://docs.astral.sh/uv/) 创建）、带 EGL 的 GPU 驱动（MuJoCo 离屏渲染），录制视频时需要 `ffmpeg`。

```bash
./setup.sh          # 用 uv 创建 .venv 并安装 requirements.txt
```

`run.sh` 默认 `MUJOCO_GL=egl`；无 GPU 或驱动异常时可改用 `MUJOCO_GL=osmesa ./run.sh`。

## 获取权重

代码运行只需要各 bundle 目录下 **3 个 ONNX 文件**（原始 `.pt` 检查点仅为留档，推理不加载）。放入后目录应为：

```
models/vggt_hierarchical_rgb64/
├── bundle.json      # 已在仓库内，含 SHA-256 清单
├── policy.onnx      # 59.8 MB
├── encoder.onnx     # 21.3 MB
└── planner.onnx     # 737.9 MB
```

| 文件 | SHA-256 |
|---|---|
| `policy.onnx` | `ba0c5c4fd06c9326bedcfd3c4f984af99d688695348ef692f0e990970021118f` |
| `encoder.onnx` | `6c14653b489f97589a1df16daf2a0d7fadd83a6595a1bd25d2fe8398721b12ad` |
| `planner.onnx` | `39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea` |

原始检查点 `model_2400.pt` 的 SHA-256 为 `e7728d6b81824ab38091dc91055e504e0dd1ade99a2260a112007d155ab30a9d`（训练 run：`runs/vggt_stairs_hierarchical_rgb64_20261027`）。导出脚本为 [scripts/export_rgb64_server.py](scripts/export_rgb64_server.py)，来源与数值对照记录见 [models/vggt_hierarchical_rgb64/import_report.json](models/vggt_hierarchical_rgb64/import_report.json)：64 组策略对照最大误差 3.46e-6，16 组 RGB→动作全链路最大误差 5.96e-7。

旧模型 `models/velocity3999/`（112×63、21 token、checkpoint `model_3999.pt`）同样只需 3 个 ONNX，其中 `planner.onnx` 与上表字节相同；`policy.onnx`/`encoder.onnx` 的校验值见其 `bundle.json`。

**启动时会逐一校验 SHA-256**，文件缺失或不匹配会直接报错，可用 `sha256sum models/vggt_hierarchical_rgb64/*.onnx` 自查。

## 网页运行

```bash
./run.sh            # 等价于 .venv/bin/python -m sim2sim.server
```

打开 **http://127.0.0.1:8080**（局域网观看可 `./run.sh --host 0.0.0.0`）。

- **自动完成楼梯测试**：静止站立 2 秒后按当前场景前进，过终点后停步观察 4 秒，页面给出判定结果；完整轨迹自动写入 `records/browser_auto_*.json`（目录不存在会自动创建）。
- **W/S** 前进/后退（后退上限 0.25 m/s），**Q/E** 左转/右转（目标 0.45 rad/s ≈ 26°/s）。按住执行、松开停止，可组合（如 W+Q）；W/S 或 Q/E 同时按下互相抵消。失去窗口焦点自动停。
- **空格/暂停** 冻结仿真时间。
- 场景可选平地、上楼、下楼、连续上下楼；每级高度 **5/10/15/20 cm** 四档，默认 15 cm、6 级、踏面 30 cm、前进指令 0.4 m/s。切换后点“重置到所选场景”生效。
- 画面右上角是与头部 RGB 同步的交叉注意力热区（softmax(QKᵀ/8)，4×7 网格），右侧可展开设置选择 query 路（任务指令/本体状态/CLS）、色标与平滑方式；平滑仅影响显示。
- 右侧小图即策略实际使用的 112×64 头部 RGB。

## 命令行验收

`sim2sim.trial` 从静止站姿做一次可复现的完整通过测试（规划种子 42），写出 JSON 报告、`npz` 关节轨迹，`--video` 另存 MP4。**通过退出码为 0**，跌倒、未到终点或停步检查失败为非零；报告路径已存在会拒绝覆盖。

```bash
# 15 cm × 3 级、踏面 30 cm、0.4 m/s（默认地形 up）
.venv/bin/python -m sim2sim.trial \
  --terrain up --height 0.15 --count 3 --tread 0.30 --speed 0.4 \
  --output records/stair_15cm.json --video

# 旧模型上楼复现
.venv/bin/python -m sim2sim.trial --bundle models/velocity3999 \
  --terrain up --height 0.10 --count 4 --tread 0.35 --speed 0.25 \
  --output records/legacy_up.json --video
```

报告内含逐步位置、双脚位置、倾角、接触与指令轨迹，`passed` 为全部判定成立。浏览器自动化测试（可选）：

```bash
uv pip install --python .venv/bin/python --no-cache -r requirements-dev.txt
.venv/bin/python -m playwright install chromium --only-shell
.venv/bin/python scripts/test_browser.py
```

## 已验证的仿真结果

以下均为 `model_2400.pt` 导出的 ONNX 驱动真实 MuJoCo 动力学的实测，从静止站姿开始，无任何根节点回写或辅助托举；原始报告与录像保存在本地 `records/`（不入库）。结果仅对应所列条件，不代表其他高度、速度、种子或真机已验证。

**5–20 cm 四档上楼**（2026-10-29，3 级、踏面 30 cm、0.4 m/s，种子 42，真实浏览器各跑一遍）：

| 每级高度 | 总时长（含起步前 2 s 与终点后 4 s） | 判定 |
|---|---:|---|
| 5 cm | 15.16 s | 通过 |
| 10 cm | 14.52 s | 通过 |
| 15 cm | 14.52 s | 通过 |
| 20 cm | 15.22 s | 通过 |

**上下楼**（2026-10-28，每侧 4 级、级高 10 cm、踏面 35 cm、0.25 m/s）：

| 场景 | 双脚过终点 | 总时长 | 末秒停步漂移 | 判定 |
|---|---:|---:|---:|---|
| 上楼 4 级 | 16.72 s | 20.72 s | 0.63 cm | 通过 |
| 下楼 4 级 | 16.74 s | 20.74 s | 0.55 cm | 通过 |
| 上楼 4 级＋下楼 4 级 | 25.56 s | 29.56 s | 1.11 cm | 通过 |

平地前进/后退/原地转向/行进中转向及松键停步均通过物理与真实浏览器测试（转向目标 0.45 rad/s）。未验证倒着上楼、台阶上原地旋转。

## 真机与上游

- `hardware/` 为真机部署入口：默认关闭额外动作目标限幅/变化率/跟踪误差中断，加入 X 键、红色按钮、遥控 SELECT 锁存阻尼停止，保留断联与数据健康检查。**尚未在真机验证**；网页 W/S/Q/E 仅控制仿真。
- 真机操作见 [docs/G1_REAL_ROBOT_GUIDE_ZH.md](docs/G1_REAL_ROBOT_GUIDE_ZH.md)、[docs/G1_RUN_COMMANDS_ZH.md](docs/G1_RUN_COMMANDS_ZH.md)、[docs/G1_RECORDING_ZH.md](docs/G1_RECORDING_ZH.md)、[docs/G1_OPTIMIZATION_STATUS_ZH.md](docs/G1_OPTIMIZATION_STATUS_ZH.md)。
- `upstream/` 保存远端部署文档与接口代码快照及来源哈希，用于核对观测布局、PD 参数与规划器行为的对应关系；许可文件见 [upstream/licenses/](upstream/licenses/)。

本仓库运行的是 **sim2sim**：执行训练所用原生 PD 目标与力矩限制。远端实机入口额外的目标投影、变化率和跟踪误差保护会改变闭环行为，这里的通过仅代表上述 MuJoCo 仿真条件，不构成实机验收。

## 常见问题

- **启动报 `Bundle checksum mismatch`**：权重文件与 `bundle.json` 清单不一致，核对上面的 SHA-256 表。
- **渲染失败 / `egl` 报错**：改用 `MUJOCO_GL=osmesa ./run.sh`。
- **页面时间比真实时间慢**：高负载下仿真实时率下降属正常，页面显示的是仿真时间。
