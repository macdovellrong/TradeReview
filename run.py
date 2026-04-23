import os
import sys


DEFAULT_WINDOWS_QT_PLATFORM = "windows:dpiawareness=1"


def configure_qt_platform(env=None, platform_name=None):
    env = os.environ if env is None else env
    platform_name = os.name if platform_name is None else platform_name

    if platform_name == "nt" and not env.get("QT_QPA_PLATFORM"):
        env["QT_QPA_PLATFORM"] = DEFAULT_WINDOWS_QT_PLATFORM

    return env.get("QT_QPA_PLATFORM")


def main(argv=None):
    configure_qt_platform()

    from PyQt6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication(sys.argv if argv is None else argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
