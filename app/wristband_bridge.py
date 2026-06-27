import json
import os
import threading
import time

try:
    import serial
except Exception:
    serial = None

try:
    import dbus
except Exception:
    dbus = None


class WristbandBridge:
    def __init__(self, config=None, log=None):
        self._config = config or {}
        self._log = log or (lambda msg: None)
        self._ser = None
        self._bus = None
        self._ble_device = None
        self._ble_char = None
        self._ble_write_candidates = []
        self._connected = False
        self._last_error = ''
        self._transport = str(self._config.get('transport', 'serial')).strip().lower() or 'serial'
        self._port = str(self._config.get('port', '/dev/rfcomm0')).strip() or '/dev/rfcomm0'
        self._baudrate = int(self._config.get('baudrate', 9600))
        self._ble_address = str(self._config.get('ble_address', '')).strip().upper()
        self._ble_service_uuid = self._normalize_uuid(self._config.get('ble_service_uuid', '0000ffe0-0000-1000-8000-00805f9b34fb'))
        self._ble_characteristic_uuid = self._normalize_uuid(self._config.get('ble_characteristic_uuid', 'auto'))
        self._ble_selected_characteristic_uuid = ''
        self._ble_chunk_size = int(self._config.get('ble_chunk_size', 20) or 20)
        self._ble_chunk_delay_s = float(self._config.get('ble_chunk_delay_s', 0.12) or 0.12)
        self._ble_write_type = str(self._config.get('ble_write_type', 'request')).strip().lower() or 'request'
        self._send_hz = float(self._config.get('send_hz', 1.0) or 1.0)
        self._reconnect_interval_s = float(self._config.get('reconnect_interval_s', 2.0) or 2.0)
        self._lock = threading.RLock()
        self._latest_packet = None
        self._latest_line = ''
        self._thread = None
        self._running = False
        self._last_connect_log_ts = 0.0

    @property
    def enabled(self):
        return bool(self._config.get('enabled', False))

    @property
    def connected(self):
        return self._connected

    @property
    def port(self):
        return self._port

    @property
    def last_error(self):
        return self._last_error

    @property
    def send_hz(self):
        return self._send_hz

    @property
    def transport(self):
        return self._transport

    def start(self):
        if not self.enabled or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name='wristband-bridge')
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._close()

    def update_packet(self, packet):
        if not self.enabled:
            return
        if not packet:
            with self._lock:
                self._latest_packet = None
                self._latest_line = ''
            return
        line = json.dumps(packet, ensure_ascii=False, separators=(',', ':'))
        with self._lock:
            self._latest_packet = dict(packet)
            self._latest_line = line + '\n'

    def status_snapshot(self):
        with self._lock:
            packet = dict(self._latest_packet or {})
        return {
            'enabled': self.enabled,
            'connected': self.connected,
            'transport': self.transport,
            'port': self.port,
            'ble_address': self._ble_address,
            'ble_characteristic_uuid': self._ble_characteristic_uuid,
            'ble_selected_characteristic_uuid': self._ble_selected_characteristic_uuid,
            'ble_write_type': self._ble_write_type,
            'ble_chunk_size': self._ble_chunk_size,
            'send_hz': self.send_hz,
            'last_error': self.last_error,
            'latest_packet': packet,
        }

    def _loop(self):
        interval = 1.0 / max(self._send_hz, 0.5)
        while self._running:
            started_at = time.time()
            if not self._ensure_connected():
                time.sleep(self._reconnect_interval_s)
                continue
            with self._lock:
                line = self._latest_line
            if line:
                try:
                    self._send_line(line)
                except Exception as exc:
                    self._last_error = str(exc)
                    self._log(f'⚠️ 手环蓝牙发送失败: {exc}')
                    self._close()
            elapsed = time.time() - started_at
            time.sleep(max(0.0, interval - elapsed))

    def _ensure_connected(self):
        if self._transport == 'ble':
            return self._ensure_ble_connected()
        return self._ensure_serial_connected()

    def _ensure_serial_connected(self):
        if serial is None:
            self._connected = False
            self._last_error = 'python serial module unavailable'
            return False
        if self._ser is not None and getattr(self._ser, 'is_open', False):
            self._connected = True
            return True
        try:
            self._ser = serial.Serial(self._port, self._baudrate, timeout=0.2)
            self._connected = True
            self._last_error = ''
            now = time.time()
            if now - self._last_connect_log_ts >= 5.0:
                self._last_connect_log_ts = now
                self._log(f'📶 手环蓝牙串口已连接: {self._port} @{self._baudrate}')
            return True
        except Exception as exc:
            self._connected = False
            self._last_error = str(exc)
            return False

    def _ensure_ble_connected(self):
        if dbus is None:
            self._connected = False
            self._last_error = 'python dbus module unavailable'
            return False
        if not self._ble_address:
            self._connected = False
            self._last_error = 'BLE address is empty'
            return False
        try:
            if self._bus is None:
                self._bus = dbus.SystemBus()
            self._ble_device = self._find_ble_device()
            if self._ble_device is None:
                self._connected = False
                self._last_error = f'BLE device not found: {self._ble_address}'
                return False
            self._connect_ble_device(self._ble_device)
            self._ble_write_candidates = self._find_ble_characteristics()
            self._ble_char = self._ble_write_candidates[0] if self._ble_write_candidates else None
            if self._ble_char is None:
                self._connected = False
                self._last_error = f'BLE characteristic not found: {self._ble_characteristic_uuid}'
                return False
            self._connected = True
            self._last_error = ''
            now = time.time()
            if now - self._last_connect_log_ts >= 5.0:
                self._last_connect_log_ts = now
                self._log(f'📶 手环 BLE 已连接: {self._ble_address} {self._ble_characteristic_uuid}')
            return True
        except Exception as exc:
            self._connected = False
            self._last_error = str(exc)
            return False

    def _send_line(self, line):
        data = line.encode('utf-8')
        if self._transport == 'ble':
            if not self._ble_write_candidates:
                raise RuntimeError('BLE characteristic is not ready')
            last_error = None
            for candidate in self._ordered_write_candidates():
                try:
                    self._send_ble_chunks(candidate, data)
                    self._ble_char = candidate['interface']
                    self._ble_selected_characteristic_uuid = candidate['uuid']
                    return
                except Exception as exc:
                    last_error = exc
                    continue
            raise last_error or RuntimeError('BLE write failed')
            return
        if not self._ser:
            raise RuntimeError('serial port is not ready')
        self._ser.write(data)
        self._ser.flush()

    def _ordered_write_candidates(self):
        preferred = []
        fallback = []
        for candidate in list(self._ble_write_candidates):
            if self._ble_characteristic_uuid != 'auto' and candidate['uuid'] == self._ble_characteristic_uuid:
                preferred.append(candidate)
            else:
                fallback.append(candidate)
        return preferred + fallback

    def _send_ble_chunks(self, candidate, data):
        char = candidate['interface']
        write_types = self._ordered_write_types(candidate)
        chunk_size = max(1, self._ble_chunk_size)
        for offset in range(0, len(data), chunk_size):
            chunk = data[offset:offset + chunk_size]
            value = dbus.Array([dbus.Byte(b) for b in chunk], signature='y')
            last_error = None
            for write_type in write_types:
                try:
                    char.WriteValue(value, {'type': write_type})
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(min(0.2, max(0.02, self._ble_chunk_delay_s)))
            if last_error is not None:
                raise last_error
            time.sleep(max(0.0, self._ble_chunk_delay_s))

    def _ordered_write_types(self, candidate):
        available = list(candidate.get('write_types') or [])
        if self._ble_write_type in available:
            return [self._ble_write_type] + [item for item in available if item != self._ble_write_type]
        return available or [self._ble_write_type]

    def _find_ble_device(self):
        objects = self._managed_objects()
        address = self._ble_address.upper()
        for path, interfaces in objects.items():
            props = interfaces.get('org.bluez.Device1')
            if not props:
                continue
            if str(props.get('Address', '')).upper() == address:
                return dbus.Interface(self._bus.get_object('org.bluez', path), 'org.bluez.Device1')
        return None

    def _connect_ble_device(self, device):
        props = dbus.Interface(device, 'org.freedesktop.DBus.Properties')
        if bool(props.Get('org.bluez.Device1', 'Connected')):
            return
        try:
            device.Connect()
        except Exception as exc:
            text = str(exc)
            if 'InProgress' not in text and 'already in progress' not in text:
                raise
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if bool(props.Get('org.bluez.Device1', 'Connected')):
                return
            time.sleep(0.1)
        raise TimeoutError(f'BLE connect timeout: {self._ble_address}')

    def _find_ble_characteristics(self):
        objects = self._managed_objects()
        address_key = self._ble_address.replace(':', '_').upper()
        service_paths = set()
        for path, interfaces in objects.items():
            if address_key not in str(path).upper():
                continue
            service = interfaces.get('org.bluez.GattService1')
            if service and self._normalize_uuid(service.get('UUID')) == self._ble_service_uuid:
                service_paths.add(str(path))
        candidates = []
        for path, interfaces in objects.items():
            if not any(str(path).startswith(service_path + '/') for service_path in service_paths):
                continue
            char = interfaces.get('org.bluez.GattCharacteristic1')
            if not char:
                continue
            uuid = self._normalize_uuid(char.get('UUID'))
            flags = {str(flag) for flag in char.get('Flags', [])}
            if self._ble_characteristic_uuid == 'auto':
                if 'write-without-response' in flags or 'write' in flags:
                    candidates.append(self._build_write_candidate(path, uuid, flags))
            elif uuid == self._ble_characteristic_uuid:
                candidates.append(self._build_write_candidate(path, uuid, flags))
        if candidates:
            self._ble_selected_characteristic_uuid = candidates[0]['uuid']
        return candidates

    def _build_write_candidate(self, path, uuid, flags):
        write_types = []
        if 'write-without-response' in flags:
            write_types.append('command')
        if 'write' in flags:
            write_types.append('request')
        if 'command' not in write_types:
            write_types.append('command')
        if 'request' not in write_types:
            write_types.append('request')
        return {
            'uuid': uuid,
            'interface': dbus.Interface(self._bus.get_object('org.bluez', path), 'org.bluez.GattCharacteristic1'),
            'write_types': write_types,
        }

    def _managed_objects(self):
        manager = dbus.Interface(self._bus.get_object('org.bluez', '/'), 'org.freedesktop.DBus.ObjectManager')
        return manager.GetManagedObjects()

    @staticmethod
    def _normalize_uuid(value):
        value = str(value or '').strip().lower()
        if value == 'auto':
            return value
        if len(value) == 4:
            return f'0000{value}-0000-1000-8000-00805f9b34fb'
        return value

    def _close(self):
        ser = self._ser
        self._ser = None
        self._ble_char = None
        self._ble_device = None
        self._ble_write_candidates = []
        self._connected = False
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
