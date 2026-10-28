import threading
from pathlib import Path

from disassembly_skill.unscrew_skill import UnscrewSkill


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "disassembly_skill"
    / "unscrew_skill.py"
)


class _AcknowledgingPublisher:
    def __init__(self, skill):
        self.skill = skill
        self.commands = []

    def publish(self, message):
        command = int(message.data)
        self.commands.append(command)
        with self.skill.data_lock:
            request_seq = int(self.skill.tool_status.get("request_seq", 0)) + 1
            command_seq = int(self.skill.tool_status.get("command_seq", 0)) + 1
            self.skill.tool_status.update(
                {
                    "connected": True,
                    "request_seq": request_seq,
                    "command_seq": command_seq,
                    "last_requested_cmd": command,
                    "last_command_ok": True,
                    "current_cmd": command,
                }
            )

    def get_subscription_count(self):
        return 1


def test_force_contact_includes_exact_threshold():
    assert UnscrewSkill._force_contact_reached(-5.0, 0.0, 5.0)
    assert UnscrewSkill._force_contact_reached(5.0, 0.0, 5.0)
    assert not UnscrewSkill._force_contact_reached(-4.99, 0.0, 5.0)


def test_unscrew_command_waits_for_fresh_serial_ack(monkeypatch):
    monkeypatch.setattr("disassembly_skill.unscrew_skill.rclpy.ok", lambda: True)
    skill = object.__new__(UnscrewSkill)
    skill.data_lock = threading.Lock()
    skill.tool_status = {
        "connected": True,
        "request_seq": 4,
        "command_seq": 3,
        "last_requested_cmd": 0,
        "last_command_ok": True,
        "current_cmd": 0,
    }
    skill.tool_pub = _AcknowledgingPublisher(skill)

    assert skill._send_tool_command(-1, "test unscrew", timeout_s=0.1, attempts=1)
    assert skill.tool_pub.commands == [-1]
    assert skill.tool_status["request_seq"] == 5
    assert skill.tool_status["command_seq"] == 4


def test_unscrew_sequence_checks_tool_before_arm_motion():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    execute_start = source.index("def execute_unscrew_command")
    execute_source = source[execute_start:]

    readiness = execute_source.index("if not self._wait_for_tool_controller()")
    reset = execute_source.index("if not self._reset_tool_for_unscrew_start()")
    approach = execute_source.index("if not self._approach_verified_hover")

    assert readiness < reset < approach


def test_single_unscrew_cycle_starts_once_and_stops_once(monkeypatch):
    monkeypatch.setattr("disassembly_skill.unscrew_skill.rclpy.ok", lambda: True)
    skill = object.__new__(UnscrewSkill)
    skill.CONFIG = {"UNSCREW_MAX_SPIN_TIME": 0.01}
    commands = []

    def send(command, _label, **_kwargs):
        commands.append(command)
        return True

    skill._send_tool_command = send

    assert skill._run_single_acknowledged_unscrew_cycle()
    assert commands == [-1, 0]


def test_single_unscrew_cycle_requires_acknowledged_stop(monkeypatch):
    monkeypatch.setattr("disassembly_skill.unscrew_skill.rclpy.ok", lambda: True)
    skill = object.__new__(UnscrewSkill)
    skill.CONFIG = {"UNSCREW_MAX_SPIN_TIME": 0.01}
    commands = []

    def send(command, _label, **_kwargs):
        commands.append(command)
        return command != 0

    skill._send_tool_command = send

    assert not skill._run_single_acknowledged_unscrew_cycle()
    assert commands == [-1, 0]


def test_assume_success_mode_retracts_after_single_cycle():
    skill = object.__new__(UnscrewSkill)
    skill.CONFIG = {
        "ASSUME_SUCCESS_AFTER_SPIN": True,
        "SCREW_GRIPPER_ENABLED": False,
    }
    events = []
    skill._wait_for_tool_controller = lambda: True
    skill._send_tool_command = lambda command, _label, **_kwargs: (
        events.append(("command", command)) or True
    )
    skill._run_single_acknowledged_unscrew_cycle = lambda: (
        events.append(("spin", None)) or True
    )
    skill._post_unscrew_style_retract = lambda _label: (
        events.append(("retract", None)) or True
    )
    skill.wait_for_arm_settled = lambda: events.append(("settled", None))

    assert skill.perform_compliant_extraction() == (True, False)
    assert events == [
        ("command", 0),
        ("spin", None),
        ("retract", None),
        ("settled", None),
    ]


def test_success_without_screw_gripper_returns_directly_home():
    skill = object.__new__(UnscrewSkill)
    skill.CONFIG = {"SCREW_GRIPPER_ENABLED": False}
    events = []
    skill._pub_state = lambda state: events.append(("state", state))
    skill._move_xarm_home = lambda: events.append(("home", None)) or True
    skill._navigate_to_bin = lambda _bin: events.append(("bin", None)) or True

    assert skill._finish_successful_unscrew(None)
    assert events == [("state", "HOMING"), ("home", None)]
