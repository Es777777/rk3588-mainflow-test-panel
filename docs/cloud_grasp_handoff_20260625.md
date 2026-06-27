# 云端辅助抓取接入说明

日期：2026-06-25

## 目标

后续要把当前“本地 RK3588 辅助抓取”模式扩展为“语音输入 -> 云端 DA3/SAM3 抓取分析 -> 返回抓取向量”的方案，同时尽量不破坏当前主程序结构、手环协议、前端状态接口和日志体系。

## 现有主流程位置

主入口与关键流程如下：

- `app/main.py`
  - `AppState`
  - `transcribe_audio()`
  - `resolve_target_object()`
  - `execute_assist_grab_target()`
  - `_build_grasp_wristband_packet()`
- `app/modes/assist_grab.py`
  - 辅助抓取模式生命周期
  - 录音结束后触发目标识别与抓取执行
- `app/modes/base.py`
  - 模式通用录音释放流程
- `app/rk3588_bridge.py`
  - 当前本地 RK3588 抓取向量运行时桥接层
- `app/wristband_bridge.py`
  - 手环/臂章向量协议发送

## 建议接入位置

不要把云端 DA3/SAM3 逻辑直接写死在 `app/modes/assist_grab.py` 里。

建议新增一层与 `app/rk3588_bridge.py` 同级的桥接模块，例如：

- `app/cloud_grasp_bridge.py`

这层负责：

1. 接收抓取提示词
2. 获取当前相机帧或所需图像输入
3. 向云端发请求
4. 解析云端返回
5. 统一输出与当前本地抓取链路一致的数据结构

## 推荐数据接口

云端抓取桥接层最终应统一产出如下字段，尽量与现有 `rk3588_bridge` 输出兼容：

```json
{
  "status": "tracking",
  "vector_active": true,
  "target_class_name": "cell phone",
  "hand_x_m": 0.0,
  "hand_y_m": 0.0,
  "hand_z_m": 0.0,
  "target_x_m": 0.0,
  "target_y_m": 0.0,
  "target_z_m": 0.0,
  "vector_x_m": 0.0,
  "vector_y_m": 0.0,
  "vector_z_m": 0.0,
  "distance_m": 0.0,
  "timestamp_s": 0.0
}
```

至少要保证这些字段可用：

- `status`
- `vector_active`
- `target_class_name`
- `vector_x_m`
- `vector_y_m`
- `vector_z_m`
- `distance_m`

原因：

- `app/main.py` 已经基于这套字段生成手环 `grasp` 包
- Web 状态接口和日志也围绕这套字段展开
- 后面如果做本地/云端切换，可以复用同一套主流程

## 推荐主流程改法

### 1. 保持模式层简单

`app/modes/assist_grab.py` 继续只做：

- 录音结束
- STT 转文本
- 目标解析
- 调用统一的抓取执行入口

不要在这里塞 HTTP 请求、图片上传、SAM3 解析和向量换算细节。

### 2. 在主程序里保留统一入口

`app/main.py` 中 `execute_assist_grab_target()` 仍然保留为统一入口，但内部改成可选择不同后端：

- 本地 `rk3588_bridge`
- 云端 `cloud_grasp_bridge`

可以通过配置切换，例如：

```json
{
  "assist_grab_backend": "rk3588"
}
```

后续切到：

```json
{
  "assist_grab_backend": "cloud"
}
```

### 3. 云端桥接层做异步/轮询更合适

因为云端分析延迟通常高于本地实时链路，所以 `cloud_grasp_bridge.py` 建议模仿现有桥接类设计：

- `start()`
- `stop()`
- `set_target_prompt()`
- `read_latest_payload()`
- `ready`
- `running`

这样主程序上层就不需要知道底层到底是本地进程、云端请求还是轮询任务。

## 语音输入建议

语音输入仍然复用当前主程序：

- `transcribe_audio()` 做语音转文字
- `resolve_target_object()` 或新增 prompt 生成器，把用户说的话整理为云端提示词

建议在主程序里新增一个“云端抓取提示词构造函数”，把原始语音文本转成更稳定的请求内容，例如：

- 用户说：`帮我抓手机`
- 发送给云端的提示词：
  - `请识别用户当前要抓取的目标物体：手机`
  - 或更明确地包含“返回目标抓取向量”

## 云端返回要求

建议提前约束云端返回格式，避免后面解析混乱。最好要求云端返回结构化 JSON，例如：

```json
{
  "ok": true,
  "target_label": "cell phone",
  "vector_x_m": -0.21,
  "vector_y_m": 0.03,
  "vector_z_m": 0.78,
  "distance_m": 0.81,
  "status": "tracking"
}
```

如果云端侧本身输出的是像素点、mask 或抓取点，而不是直接三维向量，那么推荐在 `cloud_grasp_bridge.py` 内完成二次换算，不要把换算逻辑散落到 `main.py`。

## 与手环/臂章协议的关系

只要最终仍然在主程序里形成当前 `grasp` 包格式，就不需要改手环/臂章协议层。

也就是说，云端抓取链路应尽量适配现有：

- `_build_grasp_wristband_packet()`
- `wristband_bridge.py`

而不是反过来去改协议发送层。

## 推荐新增文件

建议后续新增：

- `app/cloud_grasp_bridge.py`
- `app/cloud_grasp_client.py`

其中：

- `cloud_grasp_client.py` 负责纯 HTTP/API
- `cloud_grasp_bridge.py` 负责与主程序状态机对接

## 这次打包内建议重点参考的文件

- `app/main.py`
- `app/modes/assist_grab.py`
- `app/modes/base.py`
- `app/rk3588_bridge.py`
- `app/wristband_bridge.py`
- `app/audio/smart_recorder.py`
- `config.json`
- `donework.md`

## 当前结论

云端 DA3/SAM3 辅助抓取方案，最合适的接入点是“抓取后端桥接层”，不是模式层。

换句话说：

- `assist_grab.py` 负责触发
- `main.py` 负责统一编排
- `cloud_grasp_bridge.py` 负责真正对接云端
- `wristband_bridge.py` 继续复用当前统一协议下发向量
