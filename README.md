# RK3588 Mainflow Test Panel — 嵌入式主流程与测试面板

> 面向 RK3588 / Ubuntu 22.04 的嵌入式半成品项目，包含主流程框架和 S1/S2/S3 测试面板。当前版本用于联调、演示和挂载真实进程，后续还需要继续完善硬件适配、模型推理和业务流程。

*(截图：启动后在浏览器或原生窗口中看到的调试面板界面)*

---

## 项目状态

- 运行平台：RK3588
- 系统环境：Ubuntu 22.04
- 当前阶段：半成品 / 调试版本
- 已包含：主流程入口、Flask API、GTK/浏览器测试面板、S1/S2/S3 开关逻辑、设备自检、语音与视觉模块占位/接入代码
- 待完善：真实硬件链路、模型部署、业务进程挂载、稳定性测试

---

## 功能

| 开关 | 快捷键 | 功能 |
|------|--------|------|
| **S1** | `F1` | 模式切换：寻物模式 → 避障模式 → 导航模式 |
| **S2** | `F2`（按住） | 语音输入：按住录音，松开识别 |
| **S3** | `F3` | 重置：清空识别结果、回到第一个模式 |

- 🖥️ 原生 GTK 窗口（或自动回退到浏览器）
- 🎤 语音识别（STT）与语音合成（TTS）
- 👁️ YOLO 目标检测（寻物/避障）
- 📷 摄像头实时画面
- 🔌 设备自检（摄像头/麦克风/扬声器/网络/蓝牙）
- 📝 日志实时显示

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动
python3 main.py
```

启动后会自动打开窗口，显示调试面板。

> 如需语音识别：`pip install openai-whisper`
> 如需目标检测：`pip install ultralytics`
> 如需本地大模型：`pip install transformers`

---

## 项目结构

```
debug_panel/
├── main.py                  # 入口
├── run.sh                   # 启动脚本
├── config.json              # 配置文件
├── requirements.txt         # Python 依赖
├── setup.sh                 # 环境安装脚本
├── .gitignore
├── LICENSE                  # MIT
├── app/
│   ├── main.py              # 应用状态 & 初始化
│   ├── server.py            # Flask API 服务
│   ├── gui.py               # GTK 原生窗口
│   ├── config.py            # 配置管理
│   ├── switches.py          # S1/S2/S3 逻辑
│   ├── device_check.py      # 设备自检
│   ├── audio/
│   │   ├── stt.py           # 语音转文字
│   │   ├── tts.py           # 文字转语音
│   │   └── recorder.py      # 录音
│   ├── vision/
│   │   ├── camera.py        # 摄像头
│   │   ├── yolo_detector.py # YOLO 检测
│   │   └── intern_model.py  # Intern 模型
│   ├── modes/
│   │   ├── base.py          # 模式基类
│   │   ├── find_object.py   # 寻物模式
│   │   ├── obstacle.py      # 避障模式
│   │   └── navigation.py    # 导航模式
│   └── data/
│       └── objects.txt      # 可识别物品列表
└── static/
    ├── index.html           # 前端界面 (Vue 3)
    └── vue.global.prod.js   # Vue.js 本地副本
```

---

## 挂载真实进程

编辑 `app/modes/find_object.py` 中的对应方法，例如 S2 触发时启动进程：

```python
def on_s2_press(self):
    # 按下 S2 时启动进程
    import subprocess
    self._proc = subprocess.Popen(['your-program', '--arg'])

def on_s2_release(self):
    # 松开 S2 时停止
    if self._proc:
        self._proc.terminate()
```

---

## 配置

编辑 `config.json`：

```json
{
  "server": { "host": "0.0.0.0", "port": 5000 },
  "switches": { "s1_key": "F1", "s2_key": "F2", "s3_key": "F3" },
  "camera": { "device_id": 0 }
}
```

---

## 许可证

MIT
