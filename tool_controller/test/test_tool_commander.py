from types import SimpleNamespace

from std_msgs.msg import Int8

from tool_controller.tool_commander import ToolCommander


class _SerialRecorder:
    is_open = True

    def __init__(self):
        self.writes = []
        self.flush_count = 0

    def write(self, payload):
        self.writes.append(payload)

    def flush(self):
        self.flush_count += 1


class _Logger:
    def __init__(self):
        self.info_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def warn(self, _message):
        pass

    def error(self, _message):
        pass


def test_serial_write_updates_acknowledgement_status():
    serial = _SerialRecorder()
    logger = _Logger()
    published_status = []
    commander = SimpleNamespace(
        ser=serial,
        fake=False,
        current_cmd=0,
        command_seq=0,
        request_seq=0,
        last_requested_cmd=None,
        last_command_ok=None,
        last_error="",
        get_logger=lambda: logger,
        _is_connected=lambda: True,
        _try_connect=lambda: True,
        _publish_status=lambda: published_status.append(True),
    )

    ToolCommander.handle_cmd(commander, Int8(data=-1))

    assert serial.writes == [b"-1\n"]
    assert serial.flush_count == 1
    assert commander.request_seq == 1
    assert commander.command_seq == 1
    assert commander.current_cmd == -1
    assert commander.last_requested_cmd == -1
    assert commander.last_command_ok is True
    assert published_status == [True]
    assert "Sent UNSCREW command: -1" in logger.info_messages
