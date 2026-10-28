import ast
import threading
from pathlib import Path

import pytest

from disassembly_skill.device_config import DeviceConfig
from disassembly_skill.laptop_auto_agent import LaptopAutoAgent


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "disassembly_skill"
    / "laptop_auto_agent.py"
)
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "device_configs"
    / "laptop.yaml"
)


def _source_tree():
    return ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))


def test_sequence_contains_one_heat_sink_and_one_battery_screw():
    tree = _source_tree()
    sequence_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "AUTOMATIC_SEQUENCE"
            for target in node.targets
        )
    )
    calls = sequence_assignment.value.elts
    labels = [call.args[0].value for call in calls]

    assert labels == ["heat_sink_screw", "battery_screw"]


def test_agent_does_not_import_uf850_motion_skills():
    tree = _source_tree()
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "disassembly_skill.unscrew_skill" in imported_modules
    assert "disassembly_skill.object_pickup_skill" not in imported_modules
    assert "disassembly_skill.object_hold_skill" not in imported_modules
    assert "disassembly_skill.object_flip_skill" not in imported_modules
    assert "disassembly_skill.object_flip_drop_skill" not in imported_modules


def test_ui_uses_persistent_next_action_window():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "Action: pickup_policy" in source
    assert "_create_action_window()" in source
    assert "request_next_stage" in source
    assert "request_completion_confirmation" in source
    assert "request_action_retry" in source
    assert 'text="NEXT  →"' in source
    assert 'text="SKIP STEP"' in source
    assert 'text="CONFIRM COMPLETE"' in source
    assert 'text="RETRY ACTION"' in source
    assert "pickup_policy_wait_s" not in source
    assert "request_teleoperation" not in source
    assert "CONFIRM HEAT SINK REMOVED" not in source


def test_every_unscrew_and_pickup_stage_is_execute_or_skip_gated():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    sequence_start = source.index("def _run_sequence")
    sequence_source = source[sequence_start:]

    unscrew_gate = sequence_source.index("unscrew_decision = self._wait_for_next(")
    unscrew_skip = sequence_source.index('if unscrew_decision == "skip"')
    unscrew_action = sequence_source.index("if self._run_unscrew(target)")
    pickup_gate = sequence_source.index(
        "pickup_decision = self._wait_for_next(",
        unscrew_action,
    )
    pickup_skip = sequence_source.index('if pickup_decision == "skip"')
    pickup_action = sequence_source.index(
        "not self._run_pickup_policy(target.pickup_name, stage_number)",
        pickup_gate,
    )

    assert unscrew_gate < unscrew_skip < unscrew_action
    assert unscrew_action < pickup_gate < pickup_skip < pickup_action


def test_failed_unscrew_retries_same_stage_without_advancing():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    sequence_start = source.index("def _run_sequence")
    sequence_source = source[sequence_start:]

    retry_loop = sequence_source.index("while not self._stop_event.is_set()")
    unscrew_attempt = sequence_source.index("if self._run_unscrew(target)", retry_loop)
    retry_gate = sequence_source.index("_wait_for_unscrew_retry(stage_number, target)")
    stage_advance = sequence_source.index("stage_number += 1", retry_gate)

    assert retry_loop < unscrew_attempt < retry_gate < stage_advance


def test_pickup_waits_for_explicit_completion_confirmation():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    pickup_start = source.index("def _run_pickup_policy")
    sequence_start = source.index("def _run_sequence", pickup_start)
    pickup_source = source[pickup_start:sequence_start]

    action_message = pickup_source.index('"ACTION"')
    confirmation = pickup_source.index("_wait_for_pickup_confirmation")
    observation = pickup_source.index('"ACT"', confirmation)

    assert action_message < confirmation < observation


def test_hole_result_is_not_reported_as_successful_unscrew():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    unscrew_start = source.index("def _run_unscrew")
    pickup_start = source.index("def _run_pickup_policy", unscrew_start)
    unscrew_source = source[unscrew_start:pickup_start]

    assert 'if result is True:' in unscrew_source
    assert 'if result == "HOLE":' in unscrew_source
    assert 'result is True or result == "HOLE"' not in unscrew_source
    assert "Removal is unverified" in unscrew_source


def test_laptop_uses_bounded_tracking_for_coarse_xarm_approach():
    config = CONFIG_PATH.read_text(encoding="utf-8")

    heat_sink_step = config.index("target: heat_sink_screw")
    next_step = config.index("\n  - step:", heat_sink_step)
    heat_sink_config = config[heat_sink_step:next_step]

    assert "coarse_planner: track" in heat_sink_config
    assert "coarse_planner: direct" not in heat_sink_config
    assert "coarse_planner: direct" not in config


def test_laptop_unscrew_starts_after_confirmed_contact():
    config = DeviceConfig.load(CONFIG_PATH)
    unscrew_steps = [
        step for step in config.disassembly_sequence if step.action == "unscrew"
    ]

    assert unscrew_steps
    assert all(
        step.parameters.get("skip_final_align_on_contact") is True
        for step in unscrew_steps
    )
    assert all(
        step.parameters.get("assume_success_after_spin") is True
        for step in unscrew_steps
    )


def test_next_window_starts_disabled_and_does_not_close_between_stages():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert 'state="disabled"' in source
    assert "close_policy_window" not in source
    assert "grab_set" not in source


def test_skip_advances_without_dispatching_robot_action():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    sequence_start = source.index("def _run_sequence")
    sequence_source = source[sequence_start:]

    assert 'if unscrew_decision == "skip":' in sequence_source
    assert 'if pickup_decision == "skip":' in sequence_source
    assert "No robot or tool command was sent" in source
    assert "SEQUENCE COMPLETE — STEPS SKIPPED" in source


def test_skipping_all_stages_never_calls_robot_actions():
    agent = object.__new__(LaptopAutoAgent)
    agent.dashboard = None
    agent._stop_event = threading.Event()
    agent._skipped_stages = []
    decisions = iter(["skip", "skip", "skip", "skip"])
    agent._wait_for_next = lambda *_args, **_kwargs: next(decisions)
    agent._emit = lambda *_args, **_kwargs: None
    agent._run_unscrew = lambda *_args, **_kwargs: pytest.fail(
        "unscrew action must not run for a skipped stage"
    )
    agent._run_pickup_policy = lambda *_args, **_kwargs: pytest.fail(
        "pickup action must not run for a skipped stage"
    )

    agent._run_sequence()

    assert agent._skipped_stages == [
        "Remove heat-sink screw",
        "Pick up heat sink",
        "Remove battery screw",
        "Pick up battery",
    ]


def test_main_dashboard_does_not_duplicate_the_action_stage_banner():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "stage_indicator" not in source
    assert "set_active_stage" not in source
