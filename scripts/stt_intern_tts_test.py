#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

import requests


API_BASE = 'http://127.0.0.1:8080/v1'


def transcribe(audio_path, model='whisper-tiny', language='zh'):
    with open(audio_path, 'rb') as handle:
        response = requests.post(
            f'{API_BASE}/audio/transcriptions',
            files={'file': (os.path.basename(audio_path), handle, 'audio/wav')},
            data={'model': model, 'language': language, 'response_format': 'json'},
            timeout=300,
        )
    response.raise_for_status()
    data = response.json()
    return data.get('text', '').strip(), data


def chat(prompt, model='internvl3.5-1b'):
    response = requests.post(
        f'{API_BASE}/chat/completions',
        json={
            'model': model,
            'messages': [
                {'role': 'system', 'content': '你是一个简洁、自然、适合语音播报的中文助手。'},
                {'role': 'user', 'content': prompt},
            ],
            'stream': False,
        },
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    message = data['choices'][0]['message']['content'].strip()
    return message, data


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path


def play_wav(path, device='plughw:4,0'):
    subprocess.run(
        ['aplay', '-D', device, '-q', str(path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    parser = argparse.ArgumentParser(description='Test STT -> Intern -> TTS with local rkllama server')
    parser.add_argument('audio_path', help='Input wav file')
    parser.add_argument('--out-dir', default='/home/l/debug_panel/runtime_outputs/dialog_test')
    parser.add_argument('--play', action='store_true', help='Play generated tts wav through speaker')
    parser.add_argument('--play-device', default='plughw:4,0')
    args = parser.parse_args()

    audio_path = Path(args.audio_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stt_text, stt_raw = transcribe(str(audio_path))
    print('STT:', stt_text, flush=True)
    (out_dir / 'stt.json').write_text(json.dumps(stt_raw, ensure_ascii=False, indent=2), encoding='utf-8')

    reply, chat_raw = chat(stt_text)
    print('Intern:', reply, flush=True)
    (out_dir / 'chat.json').write_text(json.dumps(chat_raw, ensure_ascii=False, indent=2), encoding='utf-8')

    tts_path = synthesize(reply, out_dir / 'reply.wav')
    print(f'TTS saved: {tts_path}', flush=True)

    if args.play:
        play_wav(tts_path, args.play_device)
        print('TTS playback finished', flush=True)


if __name__ == '__main__':
    main()
