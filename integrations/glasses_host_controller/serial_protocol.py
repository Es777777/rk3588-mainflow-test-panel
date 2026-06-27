import struct
from dataclasses import dataclass


MAGIC = b"\xAA\x55"
VERSION = 0x01
HEADER_SIZE = 8

MSG_HELLO = 0x01
MSG_HEARTBEAT = 0x02
MSG_BUTTON_EVENT = 0x10
MSG_MODE_CHANGED = 0x11
MSG_RECORD_START = 0x20
MSG_AUDIO_UP = 0x21
MSG_RECORD_END = 0x22
MSG_PLAY_START = 0x30
MSG_AUDIO_DOWN = 0x31
MSG_PLAY_STOP = 0x32
MSG_PLAY_DONE = 0x33
MSG_SET_MODE = 0x40
MSG_BEEP = 0x41
MSG_GET_STATUS = 0x42
MSG_STATUS = 0x43
MSG_ERROR = 0xF0
MSG_LOG = 0xF1


@dataclass(frozen=True)
class Frame:
    msg_type: int
    seq: int
    payload: bytes


class FrameParser:
    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[Frame]:
        self._buffer.extend(data)
        frames: list[Frame] = []

        while True:
            magic_pos = self._buffer.find(MAGIC)
            if magic_pos < 0:
                self._buffer.clear()
                break
            if magic_pos > 0:
                del self._buffer[:magic_pos]
            if len(self._buffer) < HEADER_SIZE:
                break

            version = self._buffer[2]
            if version != VERSION:
                del self._buffer[0]
                continue

            msg_type = self._buffer[3]
            seq = struct.unpack_from("<H", self._buffer, 4)[0]
            payload_len = struct.unpack_from("<H", self._buffer, 6)[0]
            frame_len = HEADER_SIZE + payload_len + 1

            if len(self._buffer) < frame_len:
                break

            raw_frame = bytes(self._buffer[:frame_len])
            checksum = 0
            for value in raw_frame[:-1]:
                checksum ^= value

            if checksum == raw_frame[-1]:
                payload = raw_frame[HEADER_SIZE:-1]
                frames.append(Frame(msg_type=msg_type, seq=seq, payload=payload))

            del self._buffer[:frame_len]

        return frames


def build_frame(msg_type: int, seq: int, payload: bytes = b"") -> bytes:
    if len(payload) > 1024:
        raise ValueError(f"payload too large: {len(payload)}")

    header = MAGIC + bytes([VERSION, msg_type]) + struct.pack("<HH", seq & 0xFFFF, len(payload))
    raw = header + payload
    checksum = 0
    for value in raw:
        checksum ^= value
    return raw + bytes([checksum])


BUTTON_NAMES = {
    0: "MODE",
    1: "TALK",
    2: "EXTRA",
}

BUTTON_EVENT_NAMES = {
    0: "down",
    1: "up",
    2: "short",
    3: "long",
}

MODE_NAMES = {
    0: "voice_assistant",
    1: "grasp_assistant",
    2: "navigation",
}

DEVICE_STATE_NAMES = {
    0: "idle",
    1: "prompt",
    2: "recording",
    3: "host_playing",
    4: "error",
}


def parse_button_event(payload: bytes) -> dict[str, int | str]:
    if len(payload) != 8:
        raise ValueError(f"button payload must be 8 bytes, got {len(payload)}")

    button_id = payload[0]
    event_type = payload[1]
    current_mode = payload[2]
    timestamp_ms = struct.unpack_from("<I", payload, 4)[0]

    return {
        "button_id": button_id,
        "button": BUTTON_NAMES.get(button_id, f"unknown_{button_id}"),
        "event_type": event_type,
        "event": BUTTON_EVENT_NAMES.get(event_type, f"unknown_{event_type}"),
        "current_mode": current_mode,
        "mode": MODE_NAMES.get(current_mode, f"unknown_{current_mode}"),
        "timestamp_ms": timestamp_ms,
    }


def parse_mode_changed(payload: bytes) -> dict[str, int | str]:
    if len(payload) != 8:
        raise ValueError(f"mode payload must be 8 bytes, got {len(payload)}")

    mode = payload[0]
    source = payload[1]
    timestamp_ms = struct.unpack_from("<I", payload, 4)[0]

    return {
        "mode_id": mode,
        "mode": MODE_NAMES.get(mode, f"unknown_{mode}"),
        "source": "button" if source == 0 else "host",
        "timestamp_ms": timestamp_ms,
    }


def parse_status(payload: bytes) -> dict[str, int | str | bool]:
    if len(payload) < 4:
        raise ValueError(f"status payload must be at least 4 bytes, got {len(payload)}")

    mode = payload[0]
    state = payload[1]
    return {
        "mode_id": mode,
        "mode": MODE_NAMES.get(mode, f"unknown_{mode}"),
        "state_id": state,
        "state": DEVICE_STATE_NAMES.get(state, f"unknown_{state}"),
        "recording": bool(payload[2]),
        "playing": bool(payload[3]),
    }
