import threading
from types import SimpleNamespace

from disassembly_skill.unscrew_skill import UnscrewSkill


def _transform(x=0.5, y=0.2, z=1.0):
    return SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=x, y=y, z=z),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )


class _FakeTfBuffer:
    def lookup_transform(self, *_args, **_kwargs):
        return _transform()


class _FakeBackend:
    def __init__(self, skill):
        self.skill = skill
        self.tf_buffer = _FakeTfBuffer()
        self.first_target = None

    def stop_servo(self, **_kwargs):
        return True

    @staticmethod
    def _quaternion_to_rpy(*_args):
        return 0.0, 0.0, 0.0

    def move_cartesian_realtime_exotica(self, target_fn, stop_fn, **_kwargs):
        # A hole-only view must produce a spiral target instead of an abort.
        self.first_target = target_fn()
        assert self.first_target is not None

        # The next local update reacquires a centered screw. Alignment should
        # resume and complete normally.
        with self.skill.data_lock:
            self.skill.local_view = {
                "local_view": {
                    "crosshair": [320, 240],
                    "screw_heads": [{"center": [320, 240], "box": [300, 220, 340, 260]}],
                    "screws": [],
                    "holes": [],
                }
            }
        assert target_fn() is None
        assert stop_fn()
        return "DONE"


def test_hole_at_target_triggers_spiral_then_aligns(monkeypatch):
    monkeypatch.setattr("disassembly_skill.unscrew_skill.rclpy.ok", lambda: True)
    skill = object.__new__(UnscrewSkill)
    skill.data_lock = threading.Lock()
    skill.local_view = {
        "local_view": {
            "crosshair": [320, 240],
            "screw_heads": [],
            "screws": [],
            "holes": [{"center": [320, 240], "box": [300, 220, 340, 260]}],
        }
    }
    skill._align_detected_hole = False
    skill.CONFIG = {
        "WORLD_FRAME": "base_link",
        "ALIGN_RATE_HZ": 10.0,
        "XY_SPEED_ALIGN": 0.01,
        "ALIGN_MIN_SPEED": 0.002,
        "ALIGN_FULL_SPEED_ERROR_PX": 100.0,
        "ALIGN_TOLERANCE_PX": 3.0,
        "ALIGN_ERROR_FILTER_ALPHA": 0.25,
        "ALIGN_JOINT_ALPHA": 0.2,
        "ALIGN_STABLE_CYCLES": 1,
        "ALIGN_ONLY_TIMEOUT": 10.0,
        "SPIRAL_TIMEOUT": 5.0,
        "SPIRAL_SPEED": 0.002,
        "SPIRAL_GAP_MM": 2.0,
        "SPIRAL_START_DIST_MM": 2.0,
        "VISUAL_SERVO_MAX_RADIUS": 0.025,
        "ALIGN_VERIFY_TOLERANCE": 0.015,
        "ENABLE_SPIRAL_SEARCH": True,
        "ALIGN_MAX_JOINT_STEP_RAD": 0.02,
        "ALIGN_MAX_JOINT_VELOCITY_RAD_S": 0.6,
    }
    skill.wait_for_arm_settled = lambda **_kwargs: None
    skill.moveit_backend = _FakeBackend(skill)

    assert skill._perform_xy_align_search_only_exotica_legacy()
    # The first square-spiral direction is -X, with Z locked.
    assert skill.moveit_backend.first_target[0] < 0.5
    assert skill.moveit_backend.first_target[1] == 0.2
    assert skill.moveit_backend.first_target[2] == 1.0
    assert not skill._align_detected_hole
