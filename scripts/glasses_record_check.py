#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='按下并松开 ESP32 按键2 后录音并回放')
    parser.add_argument(
        '--host-controller-dir',
        default='/home/l/桌面/rk3588_glasses_sdk_20260605_diag/host_controller',
        help='rk3588_glasses_sdk_20260605_diag/host_controller 路径',
    )
    parser.add_argument('--port', default='/dev/glasses_esp32')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--button', default='EXTRA', help='按键2 对应的按钮名，默认 EXTRA')
    parser.add_argument('--out-dir', default='/home/l/debug_panel/runtime_outputs/glasses_recordings')
    args = parser.parse_args()

    host_dir = os.path.abspath(os.path.expanduser(args.host_controller_dir))
    if not os.path.isdir(host_dir):
        raise SystemExit(f'host_controller 路径不存在: {host_dir}')
    if host_dir not in sys.path:
        sys.path.insert(0, host_dir)

    from audio_utils import write_pcm_to_wav  # type: ignore
    from glasses_device import GlassesDevice  # type: ignore

    out_dir = Path(os.path.abspath(os.path.expanduser(args.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = GlassesDevice.open(port=args.port, baud=args.baud, reconnect=True)
    pcm = bytearray()
    recording = False
    last_wav = None

    def save_current():
        nonlocal last_wav
        if not pcm:
            print('未收到音频数据，跳过保存', flush=True)
            return None
        last_wav = out_dir / f"button2_check_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        write_pcm_to_wav(last_wav, bytes(pcm))
        print(f'已保存录音: {last_wav}', flush=True)
        return last_wav

    def on_audio(chunk):
        if recording:
            pcm.extend(chunk)

    def on_button_down(event):
        nonlocal recording
        pcm.clear()
        recording = True
        print(f'按键2按下，开始录音: {event}', flush=True)
        dev.start_recording()

    def on_button_up(event):
        nonlocal recording
        print(f'按键2松开，结束录音: {event}', flush=True)
        recording = False
        dev.stop_recording()

    def on_event(event):
        nonlocal recording
        if event.kind == 'record_start':
            recording = True
            print('收到 record_start', flush=True)
        elif event.kind == 'record_end':
            recording = False
            wav_path = save_current()
            if wav_path is not None:
                print('开始回放刚录好的音频到扬声器', flush=True)
                dev.play_wav(wav_path)
                print('回放完成', flush=True)
        elif event.kind in {'hello', 'log', 'status', 'disconnect'}:
            print(event.kind, event.data, flush=True)

    dev.on_audio(on_audio)
    dev.on_event(on_event)
    dev.on_button(args.button.upper(), 'down', on_button_down)
    dev.on_button(args.button.upper(), 'up', on_button_up)
    dev.start()
    dev.get_status()

    print(f'已连接设备，等待按下并松开按键2({args.button.upper()})，录音文件将保存到 {out_dir}', flush=True)
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('退出检测脚本', flush=True)
    finally:
        dev.close()


if __name__ == '__main__':
    main()
