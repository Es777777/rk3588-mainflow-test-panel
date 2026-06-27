# Done Work

1. 进入并检查项目 `/home/l/debug_panel`，梳理主程序、模式切换、相机、RK3588 运行时、眼镜 SDK 与前端状态接口。
2. 运行并排查主程序，定位主流程不稳定、模式切换与摄像头预览异常等问题。
3. 调整主流程回到以本地模式为主的可运行状态，避免 `TALK` 和 Agent 逻辑过早接管辅助抓取与导航识别。
4. 优化摄像头链路，切到 `MJPG 2560x720 30fps`，并补充相机诊断日志，减少采集卡顿与缓存积压。
5. 优化预览刷新与文件写入方式，保留处理后画面输出，并通过原子写入降低画面闪烁、半黑和撕裂问题。
6. 优化 RK3588 辅助抓取向量运行时，分离预览写出、减少阻塞，整体速度从约 `2.x~3.x fps` 提升到约 `4~6 fps`。
7. 优化 RK3588 避障运行时，加入自动拉起、较激进的采集配置和更快的轮询，使避障链路恢复连续输出能力。
8. 补充运行日志、状态接口和排查信息，方便通过 Web 面板和日志文件检修主程序、相机、RK3588 与眼镜连接状态。
9. 阅读桌面嵌入式项目修改方案，对照当前整体主程序检查设想与实现之间的差距，确认 Agent 先降级，优先跑通辅助抓取、避障和语音助手三条主线。
10. 阅读 U 盘中的手环协议文档 `/media/l/6EFC-97EA/rk3588_to_wristband_vector_protocol.md`，确认 `JDY-33` 蓝牙串口、`9600 8N1`、`JSON Lines`、`3Hz`、仅 `grasp/avoid` 两种模式等约束。
11. 新建手环串口发送模块 [app/wristband_bridge.py](/home/l/debug_panel/app/wristband_bridge.py:1)，实现串口连接、断线重连、固定频率发送、紧凑 JSON 序列化和状态快照。
12. 将手环桥接接入 [app/main.py](/home/l/debug_panel/app/main.py:1)，在 `AppState` 中统一生成 `grasp/avoid` 包，并在模式切换、辅助抓取向量更新、避障向量更新时自动刷新发送内容。
13. 在 [config.json](/home/l/debug_panel/config.json:1) 中新增 `wristband` 配置段，默认端口设为 `/dev/rfcomm0`，并保持 `9600` 波特率、`3Hz` 发送频率和自动重连配置。
14. 在 [app/server.py](/home/l/debug_panel/app/server.py:1) 中新增 `wristband` 状态输出，便于从 `/api/status` 查看手环连接状态、错误信息和最近发送包内容。
15. 通过 `py_compile` 和最小脚本验证手环协议链路，确认：
    - 辅助抓取模式会生成 `mode:"grasp"` 的有效协议包；
    - 导航识别模式会生成 `mode:"avoid"` 的有效协议包；
    - 语音助手模式会停止发送，避免残留旧向量。
16. 核对眼镜 SDK 能力，确认当前真实可用能力是按键、录音、状态和 `play_wav` 播放，不存在项目代码里假设的独立蜂鸣器硬件。
17. 明确指出当前 `Buzzer` 仅是主机侧兜底发声，不符合你们真实硬件方案；后续应统一走“眼镜语音播报 + 臂章向量”两条避障路径。
18. 核对当前臂章/手环避障向量来源，确认主链路使用的是 `LightStereo` 深度模型与 `yolo11s_ood_best_20260624.rknn` 检测模型融合输出的 `vector_x_m / vector_z_m / distance_m`。
19. 对检测模型做离线验证：
    - 直接调用 `RknnLiteYoloBackend` 对现有图片执行单帧推理；
    - 统计历史 `object_detection_60s.jsonl` 中的类别分布；
    - 测试模型侧单帧推理时延，得到约 `116ms/帧`、约 `8.6 FPS` 的纯检测速度。
20. 尝试启动实时避障运行时做整链测试，确认当前失败点是相机设备无法按配置路径打开，而不是 RKNN 模型本身加载失败。
21. 清理“蜂鸣器”硬件假设，从主流程中移除避障蜂鸣和模型就绪蜂鸣逻辑，统一回到“眼镜语音播报 + 臂章向量”两条提示路径。
22. 重做避障语音播报逻辑，在 [app/main.py](/home/l/debug_panel/app/main.py:1) 中新增障碍物中文标签、左右方位判断、距离格式化和播报节流逻辑，使自动播报和主动问询共用同一套结构化话术。
23. 删除 [config.json](/home/l/debug_panel/config.json:1) 中已不再使用的 `buzzer` 配置，避免后续误以为系统仍依赖独立蜂鸣器硬件。
24. 增强 [app/agent/local_multimodal_agent.py](/home/l/debug_panel/app/agent/local_multimodal_agent.py:1) 的提示词和结构化输出要求，补充更严格的 JSON 约束、模式名归一化、意图归一化、默认播报内容和目标物体清洗逻辑。
25. 加固 Agent JSON 解析器，支持：
    - 从 markdown 代码块中提取 JSON；
    - 清洗中英文标点、布尔值和尾逗号；
    - 从带有前缀检测摘要的混合文本中提取平衡 JSON；
    - 非 JSON 回复时自动降级到规则兜底结果。
26. 新增 [scripts/regression_smoke_test.py](/home/l/debug_panel/scripts/regression_smoke_test.py:1)，将结构化避障播报、手环 `grasp/avoid` 协议生成和 Agent 解析兜底固化成可重复执行的 smoke test。
27. 调查 `rkllama` 多模态 `500` 和空回复问题，确认根因不止一个，包括：
    - 多模态 worker 刚创建后的首个请求竞态；
    - 图文接口偶发 `200` 但 `content` 为空；
    - 模型回复前缀混入检测摘要，导致旧解析逻辑误截 JSON。
28. 在 [app/audio/rkllama_client.py](/home/l/debug_panel/app/audio/rkllama_client.py:1) 中补充多模态稳态机制，包括：
    - 图文请求自动重试；
    - 空 `content` 自动重试；
    - 图像上传前自动缩放压缩；
    - 多模态视觉通路预热；
    - 统一聊天接口内容提取逻辑。
29. 在 [app/main.py](/home/l/debug_panel/app/main.py:1) 中接入多模态启动预热流程，开机后自动选择一张可用画面进行视觉 warmup，尽量把首个图文请求的不稳定消耗在启动阶段。
30. 重新分配 RK3588 NPU 核心占用，降低 `rkllama / 深度 / YOLO` 之间的直接争抢：
    - `rkllama` 保持 `RKLLAMA_RKNN_CORE_MASK=2`
    - `LightStereo` 深度模型固定到 `core 0`
    - `YOLO / 手检测` 固定到 `core 1`
31. 调整 Agent 主链策略，不再默认强依赖第二次图文推理，而是优先使用 `Intern` 生成的画面摘要，再通过纯文本决策完成意图判断和结构化执行，从而降低多模态接口不稳定对主流程的影响。
32. 验证新的 Agent 主链在当前模式下不再误报 `should_switch_mode=true`，并修复某些候选 JSON 不是对象时触发的类型错误。
33. 多次执行 smoke test 与真实 `rkllama` 调用回归，确认：
    - `scripts/regression_smoke_test.py` 可稳定通过；
    - `Intern` 图像理解能返回精简摘要；
    - `Agent` 即使遇到图文空回复，也能退回文本决策并产出可执行结构化结果。
34. 重新梳理主程序 NPU 调度，在 [app/main.py](/home/l/debug_panel/app/main.py:1) 中加入显式的 NPU 串行闸门：
    - `rkllama` 调用期间统一串行化；
    - 调用前暂停 RK3588 实时链路并等待进程退出；
    - 增加 NPU 排空等待、恢复前延迟和一次性策略日志，避免实时链路与多模态链路同时抢占 NPU。
35. 在 [config.json](/home/l/debug_panel/config.json:1) 中新增 `npu_policy`、`vision_timeout_s`、`vision_attempts` 等稳态参数，避免把稳定性策略散落在硬编码里。
36. 加固 [app/audio/rkllama_client.py](/home/l/debug_panel/app/audio/rkllama_client.py:1)：
    - 当检测到本地 `rkllama` 进程存在但 API 不响应时，先回收旧进程再重启；
    - 将文本与图文请求分离超时/重试策略，图文链路改为更短超时，减少整机“像卡死一样一直等”的现象。
37. 重构实时 Agent 图像链路：
    - 去掉在线 `warmup_vision`，避免首轮请求在预热阶段长时间阻塞；
    - `Intern` 图像理解只尝试一次；
    - 一旦已经尝试过图像理解，后续 Agent 决策不再重复触发第二次图文请求，只走文本兜底。
38. 对 `run_server_fast.py` 做实机回归，确认：
    - `/api/status` 可正常返回，辅助抓取 RK3588 运行时能恢复到 `running=true, ready=true`；
    - `POST /api/agent_test` 不再长期卡死，当前测试已能返回 `200`，总耗时约 `22s`；
    - 调用完成后 RK3588 向量链路可以自动恢复，不再停在暂停态。
39. 检查 `debug_panel` 画面“只显示几秒然后卡住”的问题，确认这次并非主程序立刻整体卡死，而是预览链路会继续显示旧文件里的陈旧画面；同时发现系统里残留了多份 `rkllama` 进程，放大了后续不稳定风险。
40. 在 [app/server.py](/home/l/debug_panel/app/server.py:1) 中重写预览回退策略：
    - 仅允许当前模式对应的实时预览源参与回退；
    - 对 `vector_preview.jpg / obstacle_preview.jpg` 等文件增加新鲜度判断；
    - 无新帧时显式返回 `frame: null`，不再把旧图误当实时画面。
41. 在 [static/index.html](/home/l/debug_panel/static/index.html:1) 中修复前端轮询逻辑：
    - 当 `/api/camera_frame` 或 `/api/obstacle_preview` 返回空时，立即清空当前图片；
    - 不再因为前一次成功结果而把旧图一直留在屏幕上。
42. 在 [app/audio/rkllama_client.py](/home/l/debug_panel/app/audio/rkllama_client.py:1) 增加重复 `rkllama` 进程清理接口，并在 [app/main.py](/home/l/debug_panel/app/main.py:1) 启动阶段主动执行，避免反复启动后留下多个 `rkllama.server.server` 抢占 NPU。
43. 实机清理残留进程并回归验证：
    - 旧的多实例 `rkllama` 已清空；
    - 新启动的服务可正常返回 `/api/status`；
    - `/api/camera_frame` 会返回 `source=rk3588_vector` 的新鲜画面；
    - `/api/obstacle_preview` 在无新帧时返回 `frame:null`，验证了“旧图不再滞留”的修复。
44. 继续排查“帮我抓手机”链路，确认目标映射本身没有问题：
    - 本地规则可直接把“手机”映射到 YOLO 类名 `cell phone`；
    - 问题更可能出在眼镜 `MODE` 误触切换和录音流程被打断，而不是目标词表缺失。
45. 在 [app/main.py](/home/l/debug_panel/app/main.py:1) 中为眼镜按键状态机增加保护：
    - `MODE` 事件防抖；
    - 录音进行中忽略 `MODE`；
    - `TALK` 按下/松开后的短窗口内忽略 `MODE`，避免一次语音操作被抖动切到别的模式。
46. 在 [app/main.py](/home/l/debug_panel/app/main.py:1) 新增目标名展示辅助逻辑，将内部类名转换成中文别名用于播报；并在 [app/modes/assist_grab.py](/home/l/debug_panel/app/modes/assist_grab.py:1) 中改成优先播报“手机”等中文目标名，而不是 `cell phone` 这类英文类名。
47. 本地脚本回归验证：
    - `帮我抓手机` 可稳定解析成 `cell phone`；
    - 辅助抓取下发成功时播报文本已变为“开始辅助抓取手机”；
    - `MODE` 抖动、录音中 `MODE`、`TALK` 后短时间 `MODE` 均已被拦截。
48. 重新检查“视频卡住时到底是预览停了还是主程序退了”，结合 `/api/status`、进程列表与日志确认：
    - 向量输出在“帮我抓手机”场景下是正常连续更新的；
    - 但部分时刻确实出现过 `5000` 服务消失、主程序退出，不只是单纯预览停帧。
49. 纠正此前对 `rkllama` 多进程的错误判断：
    - `rkllama.server.server` 会拉起自己的 worker 子进程；
    - 这些子进程不能再被当成“重复实例”启动时清理，否则会放大不稳定性。
50. 在 [app/audio/rkllama_client.py](/home/l/debug_panel/app/audio/rkllama_client.py:1) 中调整旧进程回收逻辑，只在 `8080` 端口明确无响应时才回收，不再因为看到多个 `rkllama` 进程就误杀 worker。
51. 在 [app/main.py](/home/l/debug_panel/app/main.py:1) 中移除启动阶段对 `cleanup_duplicate_servers()` 的主动调用，避免程序一启动就干扰已存在的 `rkllama` 服务树。
52. 重写 [app/gui.py](/home/l/debug_panel/app/gui.py:1) 的本地 GUI 启动方式：
    - 不再用 daemon 线程在 GUI 进程内临时拉 Flask；
    - 改为独立子进程启动 `run_server_fast.py`，并等待 `/api/status` 就绪；
    - GUI 退出时再清理自己拉起的服务进程，从而降低“窗口还在但内置 Web 服务先死掉”的风险。
53. 回归验证：
    - 代码编译通过；
    - `run_server_fast.py` 可重新稳定监听 `5000`；
    - `5000` 与 `8080` 可同时存在，辅助抓取默认模式能重新拉起并保持可访问状态。
54. 检查辅助抓取与避障两条链路的向量输出，确认二者在实测中都曾正常产出：
    - 辅助抓取模式下“帮我抓手机”已成功下发 `cell phone`，并持续输出 `tracking` 向量；
    - 避障模式下也连续输出了 `person / car / bicycle` 等障碍向量与距离。
55. 结合日志判断“跑一段时间后停了”的现象并非单纯目标识别失效，而更像是运行时长时间没有新向量输出后缺乏自恢复，或模式切换后对应运行时未及时恢复。
56. 在 [app/rk3588_bridge.py](/home/l/debug_panel/app/rk3588_bridge.py:1) 与 [app/rk3588_obstacle_bridge.py](/home/l/debug_panel/app/rk3588_obstacle_bridge.py:1) 中新增最近一次有效 payload 时间戳记录与 `payload_age_s()` 查询接口。
57. 在 [app/modes/assist_grab.py](/home/l/debug_panel/app/modes/assist_grab.py:1) 中加入辅助抓取 watchdog：
    - 当前模式下如果向量进程未运行则自动重拉；
    - 如果超过约 `8s` 没有新的有效向量输出，则主动重启辅助抓取运行时。
58. 在 [app/modes/obstacle.py](/home/l/debug_panel/app/modes/obstacle.py:1) 中加入避障 watchdog：
    - 当前模式下如果避障进程退出则自动重拉；
    - 如果超过约 `8s` 没有新的避障向量输出，则主动重启避障运行时。
59. 对上述改动完成 `py_compile` 与 `scripts/regression_smoke_test.py` 回归，确认代码层无语法回归且核心 smoke test 继续通过。
60. 核对当前双目相机采集与实际处理分辨率，确认主程序当前采集是 `MJPG 2560x720 @ 30fps`，左右目拆分后再分别送入：
    - 深度模型 `576x320`
    - YOLO/手检测/障碍物检测 `640x640`
    并判断当前性能压力更可能来自 `MJPG` 解码、拆帧、缩放和预览编码，而不是 NPU 推理输入尺寸本身。
61. 复查辅助抓取与避障最近实测日志，确认：
    - 避障模式近期持续输出 `vector_x_m / vector_z_m / distance_m`；
    - 辅助抓取模式在目标切换到 `cell phone` 后曾连续输出有效 `tracking` 向量；
    - 预览画面上虽然会画出多个 `hand` 框，但实际用于向量计算的只有一只 hand 和一个 target。
62. 对照 [vector_pipeline.py](/home/l/桌面/rk3588_u_disk_package/python/rk3588/vector_pipeline.py:313) 复核辅助抓取目标选择逻辑，确认当前实现并不是按“手-物体相似度”选目标，而是：
    - 先按置信度与历史位置连续性选一只 hand；
    - 再以该 hand 为参考选出当前目标类的 target；
    - 最终计算 `target_xyz - hand_xyz` 作为抓取向量。
63. 复查避障播报规则与日志，确认“没有听到声音”并不是因为避障没有触发：
    - 自动避障播报此前已经多次命中；
    - 日志里能看到大量 `🔊 避障播报: ...` 记录；
    - 问题更集中在眼镜播放链路是否真的完成播报，而不是障碍检测或播报条件本身失效。
64. 深入检查眼镜 SDK 播放实现，确认旧版 [app/glasses_bridge.py](/home/l/debug_panel/app/glasses_bridge.py:1) 只要串口写成功就直接返回成功，没有等待眼镜端 `PLAY_DONE` 回执，因此主程序会把“音频已下发”误当成“已经实际播完”。
65. 直接使用眼镜 SDK 和 `/dev/glasses_esp32` 进行最小化实机验证：
    - 生成 `16kHz/单声道/16bit` 测试音频并下发到眼镜；
    - 成功收到 `PLAY_START` 与 `play_done` 事件；
    - 确认眼镜硬件链路本身是可播放的，问题不在串口或眼镜扬声器完全失效。
66. 重构 [app/glasses_bridge.py](/home/l/debug_panel/app/glasses_bridge.py:1) 的播放确认机制：
    - 新增播放状态锁、`PLAY_DONE` 等待和超时处理；
    - 仅当眼镜端明确返回 `play_done` 时才视为播报成功；
    - 超时或断开时写入明确日志，便于后续区分“已发送未播完”和“真实播放成功”。
67. 使用主程序同一套 `GlassesBridge + rkllama_client` 再次做语音播报验证，确认当前链路已经可以稳定得到 `✅ Glasses SDK 播放完成`，从而证明“主程序走眼镜播报”这条路径在现版本是可用的。
68. 按最新要求微调避障自动播报策略，在 [app/main.py](/home/l/debug_panel/app/main.py:859) 中将自动避障播报改为更保守的节流：
    - 仅在障碍信息变化且距离小于 `2m` 时考虑播报；
    - 自动播报的最小时间间隔统一拉长到 `10s`，降低连续提醒频率。
69. 按最新要求调整眼镜播报音量，在 [app/audio/rkllama_client.py](/home/l/debug_panel/app/audio/rkllama_client.py:363) 中将眼镜侧播放音量系数从 `0.7` 降到 `0.49`，相当于在此前基础上再下调约 `30%`。
70. 对今天早上的播放链路与播报节流改动完成 `py_compile` 复查，确认：
    - 眼镜播报确认逻辑无语法错误；
    - 避障 `10s` 节流逻辑已生效；
    - 眼镜侧降音量配置已生效。
