# Debug Panel — S1/S2/S3 调试面板 / RK3588 辅助抓取前端

> 一个带图形界面的调试面板，支持三个开关（S1/S2/S3）的模式切换、语音输入、设备自检，并可调用桌面上的 `rk3588_u_disk_package` 输出本地 YOLO + 深度三维向量。

*(截图：启动后在浏览器或原生窗口中看到的调试面板界面)*

---

## 功能

| 开关 | 快捷键 | 功能 |
|------|--------|------|
| **S1** | `F1` | 模式切换：辅助抓取模式 → 导航识别模式 → 语音助手 |
| **S2** | `F2`（按住） | 语音输入：按住录音，松开识别 |
| **S3** | `F3` | 重置：清空识别结果、回到第一个模式 |

- 🤖 辅助抓取模式：调用 `~/桌面/rk3588_u_disk_package` 运行本地 RK3588 向量模型
- 🖥️ 原生 GTK 窗口（或自动回退到浏览器）
- 🎤 语音识别（STT）与语音合成（TTS）
- 👁️ YOLO 目标检测与画面理解
- 📷 摄像头实时画面
- 🔌 设备自检（摄像头/麦克风/扬声器/网络/蓝牙）
- 🕹️ 可选眼镜侧 Glasses SDK 按键监听，开机默认进入辅助抓取模式，首次按钮触发前保持等待态
- 🔔 本地模型首次产生向量输出后触发蜂鸣器提示
- 🧠 本地多模态 agent：语音 + 画面 + 检测 + 运行时状态统一决策
- 👓 Intern 本地图像理解：为 agent 提供画面摘要
- 🔁 自动模式切换与任务接棒：一句话可直接切模式并继续执行
- 🖼️ RK3588 独占摄像头时，主程序会抓取一张初始预览帧并回退展示缓存图
- 📝 日志实时显示

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动原生窗口
python3 main.py
```

启动后会自动打开窗口，显示调试面板。

其他常用入口：

```bash
# 原生窗口启动器（带 torchvision shim）
python3 main_launcher.py

# 直接启动 Web 服务
python3 run_server_fast.py

# 静音 Web 服务
python3 run_server_silent.py

# 静音主程序 + Flask
python3 run_main.py

# 静音视觉联调入口
python3 start_visual_test.py
```

> 如需语音识别：`pip install openai-whisper`
> 如需目标检测：`pip install ultralytics`
> 如需本地大模型：`pip install transformers`
> 如需眼镜侧串口按键：`pip install pyserial`

---

## 项目结构

```
debug_panel/
├── main.py                  # 入口
├── run.sh                   # 启动脚本
├── main_launcher.py         # 原生窗口启动器（带 torchvision shim）
├── run_main.py              # 静音主程序 + Flask
├── run_server_fast.py       # 快速启动 Web 服务
├── run_server_silent.py     # 静音 Web 服务
├── start_visual_test.py     # 视觉联调入口
├── config.json              # 主程序配置文件
├── configs/
│   └── rk3588_vector_bridge_fast.json   # 调用桌面 RK3588 向量运行时的桥接配置
├── requirements.txt         # Python 依赖
├── setup.sh                 # 环境安装脚本
├── .gitignore
├── LICENSE                  # MIT
├── app/
│   ├── main.py              # 应用状态 & 初始化
│   ├── server.py            # Flask API 服务
│   ├── gui.py               # GTK 原生窗口
│   ├── agent/
│   │   └── local_multimodal_agent.py  # 本地多模态 agent
│   ├── config.py            # 配置管理
│   ├── rk3588_bridge.py     # 桌面 RK3588 向量运行时桥接
│   ├── rk3588_obstacle_bridge.py # RK3588 避障运行时桥接
│   ├── glasses_bridge.py    # 眼镜串口 / 按键 / 播放桥接
│   ├── buzzer.py            # 蜂鸣器提示
│   ├── switches.py          # S1/S2/S3 逻辑
│   ├── device_check.py      # 设备自检
│   ├── audio/
│   │   ├── stt.py           # 语音转文字
│   │   ├── tts.py           # 文字转语音
│   │   ├── recorder.py      # 本地录音
│   │   ├── glasses_recorder.py # 眼镜录音
│   │   └── smart_recorder.py   # 自动选择录音后端
│   ├── vision/
│   │   ├── camera.py        # 摄像头
│   │   ├── yolo_detector.py # YOLO 检测
│   │   └── intern_model.py  # Intern 模型
│   ├── modes/
│   │   ├── base.py          # 模式基类
│   │   ├── assist_grab.py   # 辅助抓取模式
│   │   ├── obstacle.py      # 导航识别模式
│   │   └── voice_assistant.py # 语音助手模式
│   └── data/
│       └── objects.txt      # 可识别物品列表
├── runtime_outputs/
│   └── rk3588/              # RK3588 向量输出目录
└── static/
    ├── index.html           # 前端界面 (Vue 3)
    └── vue.global.prod.js   # Vue.js 本地副本
```

---

## Agent 主链路

现在主程序不是“每个 mode 自己拼一套语音逻辑”，而是统一走 `AppState`：

1. `S2` 按下后由 `SmartRecorder` 自动选择眼镜麦克风或本地麦克风录音。
2. `S2` 松开后统一转写语音，再抓当前画面。
3. 若有画面，`InternModel` 先给出简短场景摘要；本地 YOLO 同时提供检测结果。
4. `LocalMultimodalAgent` 接收：
   - 用户语音
   - 当前模式
   - 当前图像
   - YOLO 检测
   - Intern 摘要
   - 辅助抓取/避障运行时向量
   - 连续任务上下文
5. Agent 输出严格 JSON，主程序只认结构化字段：
   - `intent`
   - `target_object`
   - `should_switch_mode`
   - `mode_name`
   - `speak_text`
6. `AppState.execute_agent_result()` 负责三件事：
   - 判断是否切模式
   - 维护连续任务上下文
   - 生成待执行的 `pending_agent_action`
7. 切到目标模式后，由对应 mode 在 `on_enter()` 里自动 `consume_pending_agent_action()`，继续完成真正动作：
   - 辅助抓取：下发目标到 RK3588 向量进程
   - 导航识别：读取最近障碍物向量并回答
   - 语音助手：走 Intern 场景理解链路

这意味着现在一句“帮我抓手机”不再只是得到一段回复，而是会经过：

`STT -> 画面/检测/摘要 -> Agent JSON -> 自动切到辅助抓取模式 -> 下发目标 -> RK3588 输出向量`

导航识别问答同理会经过：

`STT -> Agent JSON -> 自动切到导航识别模式 -> 读取障碍物向量/距离 -> 播报`

---

## 配置

编辑 `config.json`：

```json
{
  "server": { "host": "0.0.0.0", "port": 5000 },
  "switches": { "s1_key": "F1", "s2_key": "F2", "s3_key": "F3" },
  "camera": { "device_id": 0 },
  "rk3588_runtime": {
    "enabled": true,
    "config_path": "/home/l/debug_panel/configs/rk3588_vector_bridge_fast.json"
  },
  "glasses_sdk": {
    "enabled": true,
    "host_controller_dir": "/home/l/桌面/final_glasses_sdk/host_controller",
    "port": "/dev/glasses_esp32",
    "baudrate": 115200,
    "auto_enter_assist_mode": true
  }
}
```

## RK3588 向量输出位置

主程序默认调用桌面目录里的：

```text
/home/l/桌面/rk3588_u_disk_package
```

但为了方便主程序读取，桥接配置会把向量输入输出文件落到当前项目：

```text
/home/l/debug_panel/runtime_outputs/rk3588/
```

重点文件：

- 最新滚动向量输出：`/home/l/debug_panel/runtime_outputs/rk3588/vector_output_fast_latest.jsonl`
- 完整 JSONL 历史：`/home/l/debug_panel/runtime_outputs/rk3588/vector_output_fast.jsonl`
- 完整 CSV 历史：`/home/l/debug_panel/runtime_outputs/rk3588/vector_output_fast.csv`
- 主程序写给 RK3588 运行时的目标输入：`/home/l/debug_panel/runtime_outputs/rk3588/target_input.json`

`vector_output_fast_latest.jsonl` 每行是一条 JSON 记录，主程序面板中的“辅助抓取向量”卡片读取的就是这里的最新结果。

## 预览帧来源

当本地相机由 RK3588 运行时占用时，主程序不会持续打开 USB 摄像头，而是在初始化时抓取一张单帧预览并保存到：

```text
/home/l/debug_panel/runtime_outputs/agent/latest_scene.jpg
```

Web 端 `/api/camera_frame` 的取图顺序是：

- 本地主程序实时相机帧
- RK3588 避障预览图
- `runtime_outputs/agent/latest_scene.jpg` 缓存预览图

## 开机自启

执行：

```bash
./install.sh
```

安装脚本会同时生成：

- 桌面快捷方式：`~/桌面/debug-panel.desktop`
- 应用菜单项：`~/.local/share/applications/debug-panel.desktop`
- 桌面会话开机自启：`~/.config/autostart/debug-panel.desktop`

说明：

- 这是“桌面会话自启动”方案，适合这套带 GUI 的主程序
- 如果主板设置了自动登录桌面，开机后会自动启动本程序

---

## 许可证

MIT
