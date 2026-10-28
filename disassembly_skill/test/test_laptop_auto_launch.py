import ast
from pathlib import Path


LAUNCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "launch"
    / "laptop_auto_disassembly.launch.py"
)


def test_complete_launch_contains_required_runtime_components():
    source = LAUNCH_PATH.read_text(encoding="utf-8")
    ast.parse(source)

    required = {
        "disassembly_system.launch.py",
        'executable="vuer_quest_bridge"',
        'executable="wait_for_exotica_ready"',
        'executable="exotica_arm_teleop"',
        '"enable_xarm5": False',
        '"uf850.track_pitch": True',
        '"uf850.track_roll": True',
        'default_value="false",\n                description="Stream camera panels to Quest',
        'executable="laptop_auto_agent"',
    }

    missing = sorted(item for item in required if item not in source)
    assert not missing, f"missing launch components: {missing}"


def test_complete_launch_starts_agent_only_after_exotica_is_ready():
    source = LAUNCH_PATH.read_text(encoding="utf-8")

    base_position = source.index("base_system,")
    handler_position = source.index("readiness_handler,")
    gate_position = source.index("readiness_gate,")

    assert base_position < handler_position < gate_position
    assert "target_action=wait_for_exotica" in source
    assert "if event.returncode != 0" in source
    assert "laptop_agent," in source[source.index("def start_ready_consumers"):]
