# Host Controller Step 1

This first host test opens the ESP32-C3 serial port and waits for `MSG_HELLO`.

## Install dependency

```powershell
python -m pip install pyserial
```

## Run

```powershell
python host_controller\main.py --port COM5
```

If `--port` is omitted, the script lists available serial ports and tries the first one.
