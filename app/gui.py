import threading
import webbrowser
import time
import urllib.request
import os


def start_flask():
    def _run():
        from app.server import run_server
        run_server()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    for _ in range(30):
        try:
            urllib.request.urlopen('http://localhost:5000/api/status', timeout=2)
            return
        except Exception:
            time.sleep(1)


def run_native():
    start_flask()

    try:
        import gi
        gi.require_version('Gtk', '3.0')
        gi.require_version('WebKit2', '4.0')
        from gi.repository import Gtk, WebKit2

        win = Gtk.Window(title='Debug Panel')
        win.set_default_size(700, 800)
        win.set_position(Gtk.WindowPosition.CENTER)
        win.set_resizable(True)

        web = WebKit2.WebView()
        web.get_settings().set_enable_javascript(True)
        web.load_uri('http://localhost:5000')

        win.add(web)

        win.connect('destroy', Gtk.main_quit)
        win.show_all()
        Gtk.main()
    except Exception as e:
        print(f'窗口失败: {e}')
        webbrowser.open('http://localhost:5000')
        while True:
            time.sleep(60)
