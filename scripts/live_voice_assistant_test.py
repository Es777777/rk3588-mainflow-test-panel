#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import requests


API_BASE = 'http://127.0.0.1:8080/v1'
HOST_CONTROLLER_DIR = '/home/l/桌面/final_glasses_sdk/host_controller'
RECORDINGS_DIR = Path('/home/l/debug_panel/runtime_outputs/live_voice_assistant')


def ensure_host_controller():
    if HOST_CONTROLLER_DIR not in sys.path:
        sys.path.insert(0, HOST_CONTROLLER_DIR)


def transcribe(audio_path, model='whisper-tiny', language='zh'):
    with open(audio_path, 'rb') as handle:
        response = requests.post(
            f'{API_BASE}/audio/transcriptions',
            files={'file': (os.path.basename(audio_path), handle, 'audio/wav')},
            data={'model': model, 'language': language, 'response_format': 'json'},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


def chat(prompt, model='internvl3.5-1b'):
    response = requests.post(
        f'{API_BASE}/chat/completions',
        json={
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': '你是一个中文语音助手。请直接回答用户问题，简短、自然、口语化，控制在两三句话，不要输出说明、步骤、免责声明或舞台提示。',
                },
                {'role': 'user', 'content': prompt},
            ],
            'stream': False,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def synthesize(text, output_path, model='zh_CN-huayan-medium'):
    response = requests.post(
        f'{API_BASE}/audio/speech',
        json={
            'model': model,
            'input': text,
            'response_format': 'wav',
            'speed': 1.0,
        },
        timeout=300,
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)


def convert_wav_to_16k(src_path, dst_path):
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i',
            str(src_path),
            '-ar',
            '16000',
            '-ac',
            '1',
            '-sample_fmt',
            's16',
            str(dst_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def describe_wav(path):
    with wave.open(str(path), 'rb') as handle:
        return {
            'channels': handle.getnchannels(),
            'sample_width': handle.getsampwidth(),
            'frame_rate': handle.getframerate(),
            'frames': handle.getnframes(),
        }


def run_session():
    ensure_host_controller()
    from audio_utils import write_pcm_to_wav  # type: ignore
    from glasses_device import GlassesDevice  # type: ignore

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    dev = GlassesDevice.open('/dev/glasses_esp32')
    pcm = bytearray()
    recording = False
    hello_seen = False

    def on_audio(chunk):
        if recording:
            pcm.extend(chunk)

    def on_event(event):
        nonlocal hello_seen, recording
        if event.kind == 'hello':
            if not hello_seen:
                hello_seen = True
                print('设备已连接，按住 TALK 说话，松手后自动回复。', flush=True)
        elif event.kind == 'button' and isinstance(event.data, dict):
            if event.data.get('button') == 'TALK' and event.data.get('event') == 'down':
                print('开始录音，请说话...', flush=True)
            elif event.data.get('button') == 'TALK' and event.data.get('event') == 'up':
                print('录音结束，处理中...', flush=True)
        elif event.kind == 'status':
            print(f"状态: {event.data}", flush=True)
        elif event.kind == 'play_done':
            print('设备回传: 播放完成', flush=True)
        elif event.kind == 'disconnect':
            print(f'设备断开: {event.data}', flush=True)
        elif event.kind == 'log':
            print(f'设备日志: {event.data}', flush=True)
        elif event.kind == 'record_start':
            pcm.clear()
            recording = True
        elif event.kind == 'record_end':
            recording = False
            if not pcm:
                print('没有收到音频数据，请重试。', flush=True)
                return

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            wav_path = RECORDINGS_DIR / f'record_{ts}.wav'
            stt_path = RECORDINGS_DIR / f'stt_{ts}.json'
            chat_path = RECORDINGS_DIR / f'chat_{ts}.json'
            reply_wav_path = RECORDINGS_DIR / f'reply_{ts}.wav'
            reply_16k_path = RECORDINGS_DIR / f'reply_{ts}_16k.wav'
            write_pcm_to_wav(wav_path, bytes(pcm))
            print(f'录音已保存: {wav_path}', flush=True)

            stt_data = transcribe(str(wav_path))
            stt_text = stt_data.get('text', '').strip()
            stt_path.write_text(json.dumps(stt_data, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'STT: {stt_text or "<空>"}', flush=True)
            if not stt_text:
                return

            chat_data = chat(stt_text)
            reply_text = chat_data['choices'][0]['message']['content'].strip()
            chat_path.write_text(json.dumps(chat_data, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'Intern: {reply_text}', flush=True)

            synthesize(reply_text, reply_wav_path)
            convert_wav_to_16k(reply_wav_path, reply_16k_path)
            wav_info = describe_wav(reply_16k_path)
            print(
                f"TTS 16k文件: {reply_16k_path} "
                f"({wav_info['frame_rate']}Hz, {wav_info['channels']}ch, {wav_info['sample_width'] * 8}bit)",
                flush=True,
            )
            print('开始语音播报...', flush=True)
            dev.play_wav(reply_16k_path)
            print('播报完成，等待下一次说话。', flush=True)
            pcm.clear()

    dev.on_audio(on_audio)
    dev.on_event(on_event)
    dev.start()
    dev.get_status()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('测试结束。', flush=True)
    finally:
        dev.close()


if __name__ == '__main__':
    run_session()
