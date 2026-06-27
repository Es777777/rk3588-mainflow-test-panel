from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import serial
import serial.tools.list_ports

from audio_utils import AUDIO_CHUNK_BYTES, read_wav_as_pcm
from serial_protocol import (
    MSG_AUDIO_DOWN,
    MSG_AUDIO_UP,
    MSG_BEEP,
    MSG_BUTTON_EVENT,
    MSG_GET_STATUS,
    MSG_HELLO,
    MSG_LOG,
    MSG_MODE_CHANGED,
    MSG_PLAY_DONE,
    MSG_PLAY_START,
    MSG_PLAY_STOP,
    MSG_RECORD_END,
    MSG_RECORD_START,
    MSG_SET_MODE,
    MSG_STATUS,
    Frame,
    FrameParser,
    build_frame,
    parse_button_event,
    parse_mode_changed,
    parse_status,
)


ESP32_C3_USB_VID = 0x303A
ESP32_C3_USB_PID = 0x1001


@dataclass(frozen=True)
class DeviceEvent:
    kind: str
    data: dict | bytes | str | None = None


ButtonCallback = Callable[[dict], None]
AudioCallback = Callable[[bytes], None]
EventCallback = Callable[[DeviceEvent], None]


class GlassesDevice:
    def __init__(self, port: str | None = None, baud: int = 921600, reconnect: bool = True) -> None:
        self.port = port
        self.baud = baud
        self.reconnect = reconnect
        self._serial: serial.Serial | None = None
        self._parser = FrameParser()
        self._seq = 0
        self._running = False
        self._rx_thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._button_handlers: dict[tuple[str, str], list[ButtonCallback]] = {}
        self._event_handlers: list[EventCallback] = []
        self._audio_handlers: list[AudioCallback] = []

    @staticmethod
    def find_port() -> str:
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if port.vid == ESP32_C3_USB_VID and port.pid == ESP32_C3_USB_PID:
                return port.device
        for port in ports:
            text = f"{port.description} {port.hwid}".lower()
            if "esp" in text or "usb serial" in text or "ttyacm" in port.device.lower():
                return port.device
        raise RuntimeError("ESP32-C3 glasses controller was not found")

    @classmethod
    def open(cls, port: str | None = None, baud: int = 921600, reconnect: bool = True) -> "GlassesDevice":
        dev = cls(port=port, baud=baud, reconnect=reconnect)
        dev.connect()
        return dev

    def connect(self) -> None:
        if self._serial is not None and self._serial.is_open:
            return
        selected_port = self.port or self.find_port()
        self._serial = serial.Serial(selected_port, self.baud, timeout=0.1)
        self.port = selected_port

    def start(self) -> None:
        self.connect()
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, name="glasses-rx", daemon=True)
        self._rx_thread.start()

    def run_forever(self) -> None:
        if not self._running:
            self.start()
        try:
            while self._running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.close()

    def close(self) -> None:
        self._running = False
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def on_button(self, button: str, event: str, callback: ButtonCallback) -> None:
        key = (button.upper(), event.lower())
        self._button_handlers.setdefault(key, []).append(callback)

    def on_audio(self, callback: AudioCallback) -> None:
        self._audio_handlers.append(callback)

    def on_event(self, callback: EventCallback) -> None:
        self._event_handlers.append(callback)

    def set_mode(self, mode: int) -> None:
        if not 0 <= mode <= 2:
            raise ValueError("mode must be 0, 1, or 2")
        self.send(MSG_SET_MODE, bytes([mode]))

    def get_status(self) -> None:
        self.send(MSG_GET_STATUS)

    def start_recording(self) -> None:
        self.send(MSG_RECORD_START)

    def stop_recording(self) -> None:
        self.send(MSG_RECORD_END)

    def play_prompt_set(self) -> None:
        self.send(MSG_BEEP, b"\x00")

    def beep(self, count: int = 1) -> None:
        self.send(MSG_BEEP, bytes([max(1, min(255, count))]))

    def play_pcm(self, pcm: bytes, chunk_delay: float = 0.002, start_delay: float = 0.1) -> None:
        self.send(MSG_PLAY_START)
        if start_delay > 0:
            time.sleep(start_delay)
        for offset in range(0, len(pcm), AUDIO_CHUNK_BYTES):
            self.send(MSG_AUDIO_DOWN, pcm[offset : offset + AUDIO_CHUNK_BYTES])
            if chunk_delay > 0:
                time.sleep(chunk_delay)
        self.send(MSG_PLAY_STOP)

    def play_wav(self, path: str | Path, chunk_delay: float = 0.002, start_delay: float = 0.1) -> None:
        self.play_pcm(read_wav_as_pcm(path), chunk_delay=chunk_delay, start_delay=start_delay)

    def send(self, msg_type: int, payload: bytes = b"") -> None:
        if self._serial is None or not self._serial.is_open:
            self.connect()
        assert self._serial is not None
        with self._write_lock:
            self._serial.write(build_frame(msg_type, self._seq, payload))
            self._serial.flush()
            self._seq = (self._seq + 1) & 0xFFFF

    def _rx_loop(self) -> None:
        while self._running:
            try:
                self.connect()
                assert self._serial is not None
                data = self._serial.read(4096)
                if not data:
                    continue
                for frame in self._parser.feed(data):
                    self._handle_frame(frame)
            except Exception as exc:
                self._emit(DeviceEvent("disconnect", str(exc)))
                if not self.reconnect:
                    self._running = False
                    break
                if self._serial is not None:
                    try:
                        self._serial.close()
                    except Exception:
                        pass
                    self._serial = None
                time.sleep(1.0)

    def _handle_frame(self, frame: Frame) -> None:
        if frame.msg_type == MSG_HELLO:
            self._emit(DeviceEvent("hello", frame.payload.decode("utf-8", errors="replace")))
            return
        if frame.msg_type == MSG_BUTTON_EVENT:
            event = parse_button_event(frame.payload)
            self._emit(DeviceEvent("button", event))
            key = (str(event["button"]).upper(), str(event["event"]).lower())
            for callback in self._button_handlers.get(key, []):
                callback(event)
            return
        if frame.msg_type == MSG_MODE_CHANGED:
            self._emit(DeviceEvent("mode_changed", parse_mode_changed(frame.payload)))
            return
        if frame.msg_type == MSG_RECORD_START:
            self._emit(DeviceEvent("record_start", None))
            return
        if frame.msg_type == MSG_AUDIO_UP:
            self._emit(DeviceEvent("audio_up", {"seq": frame.seq, "bytes": len(frame.payload)}))
            for callback in self._audio_handlers:
                callback(frame.payload)
            return
        if frame.msg_type == MSG_RECORD_END:
            self._emit(DeviceEvent("record_end", None))
            return
        if frame.msg_type == MSG_PLAY_DONE:
            self._emit(DeviceEvent("play_done", None))
            return
        if frame.msg_type == MSG_STATUS:
            self._emit(DeviceEvent("status", parse_status(frame.payload)))
            return
        if frame.msg_type == MSG_LOG:
            self._emit(DeviceEvent("log", frame.payload.decode("utf-8", errors="replace")))
            return
        self._emit(DeviceEvent("frame", {"type": frame.msg_type, "seq": frame.seq, "payload": frame.payload}))

    def _emit(self, event: DeviceEvent) -> None:
        for callback in self._event_handlers:
            callback(event)
