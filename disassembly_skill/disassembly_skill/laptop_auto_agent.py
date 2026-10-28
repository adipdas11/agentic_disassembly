#!/usr/bin/env python3
"""Laptop disassembly flow with a Next-gated ReAct dashboard.

The node commands the xArm5-backed ``UnscrewSkill``. Every procedure stage is
released from a persistent action-control window, so launching the stack alone
never starts a robot action. UF850 pickup stages remain visual policy events;
this module does not construct a UF850 motion backend or publish UF850 motion
commands.
"""

from __future__ import annotations

import copy
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import scrolledtext
from typing import Callable

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Int8

from disassembly_skill.device_config import DeviceConfig
from disassembly_skill.unscrew_skill import UnscrewSkill


DEFAULT_CONFIG = (
    Path(get_package_share_directory("disassembly_skill"))
    / "config"
    / "device_configs"
    / "laptop.yaml"
)

NODE_COLOURS = {
    "VISION": "#FFD700",
    "THINK": "#9370DB",
    "PLAN": "#1E90FF",
    "ACTION": "#FFA500",
    "ACT": "#32CD32",
}


@dataclass(frozen=True)
class UnscrewTarget:
    label: str
    display_name: str
    pickup_name: str


# Exactly one invocation is made for each entry.
AUTOMATIC_SEQUENCE = (
    UnscrewTarget("heat_sink_screw", "heat-sink screw", "heat sink"),
    UnscrewTarget("battery_screw", "battery screw", "battery"),
)


class ReActDashboard:
    """Dashboard matching the existing master-agent monitor."""

    def __init__(self, on_stop: Callable[[], None]):
        self._on_stop = on_stop
        self._next_event: threading.Event | None = None
        self._action_window: tk.Toplevel | None = None
        self._pending_action_title = ""
        self._pending_action_mode = "start"
        self._stage_decision: str | None = None

        self.root = tk.Tk()
        self.root.title("🧠 LangGraph Autonomous Brain Monitor")
        self.root.geometry("850x650")
        self.root.minsize(760, 560)
        self.root.configure(bg="#1e1e1e")
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        title = tk.Label(
            self.root,
            text="Agent Execution State",
            fg="white",
            bg="#1e1e1e",
            font=("Arial", 16, "bold"),
        )
        title.pack(pady=(10, 4))

        self.status_label = tk.Label(
            self.root,
            text="INITIALISING ROBOT INTERFACES",
            fg="#7ecfff",
            bg="#1e1e1e",
            font=("Arial", 10, "bold"),
        )
        self.status_label.pack(pady=(0, 8))

        node_frame = tk.Frame(self.root, bg="#1e1e1e")
        node_frame.pack(pady=6)
        self.node_labels: dict[str, tk.Label] = {}
        for node_name in ("VISION", "THINK", "PLAN", "ACTION", "ACT"):
            label = tk.Label(
                node_frame,
                text=node_name,
                width=12,
                height=2,
                font=("Arial", 12, "bold"),
                bg="#333333",
                fg="gray",
                relief="ridge",
                borderwidth=2,
            )
            label.pack(side=tk.LEFT, padx=5)
            self.node_labels[node_name] = label

        self.log_area = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            bg="#000000",
            fg="#d4d4d4",
            insertbackground="white",
            font=("Consolas", 11),
            borderwidth=0,
        )
        self.log_area.pack(expand=True, fill="both", padx=20, pady=(14, 10))
        for node_name, colour in NODE_COLOURS.items():
            self.log_area.tag_config(node_name, foreground=colour)
        self.log_area.insert(
            tk.END,
            "Waiting for vision snapshot and mission trigger...\n\n",
        )
        self.log_area.configure(state="disabled")

        controls = tk.Frame(self.root, bg="#151515", pady=9)
        controls.pack(fill="x", side="bottom")
        self.stop_button = tk.Button(
            controls,
            text="STOP EXECUTION",
            command=self._on_stop,
            bg="#5b2020",
            fg="#ffb3b3",
            activebackground="#7a2929",
            activeforeground="white",
            relief="flat",
            font=("Arial", 10, "bold"),
            padx=18,
            pady=8,
        )
        self.stop_button.pack(side="right", padx=20)

        # Keep a separate, compact action controller visible for the complete
        # procedure. It is intentionally non-modal so the dashboard Stop
        # control and the other ROS UIs remain accessible.
        self._create_action_window()

    def emit(self, node_name: str, text: str) -> None:
        node_name = node_name.upper()

        def update() -> None:
            for name, label in self.node_labels.items():
                label.configure(bg="#333333", fg="gray")
            if node_name in self.node_labels:
                foreground = "black" if node_name == "VISION" else "white"
                self.node_labels[node_name].configure(
                    bg=NODE_COLOURS[node_name],
                    fg=foreground,
                )
            self.log_area.configure(state="normal")
            self.log_area.insert(
                tk.END,
                f"[{node_name}] {text}\n\n",
                node_name if node_name in NODE_COLOURS else None,
            )
            self.log_area.see(tk.END)
            self.log_area.configure(state="disabled")

        self.root.after(0, update)

    def set_status(self, text: str, colour: str = "#7ecfff") -> None:
        self.root.after(
            0,
            lambda: self.status_label.configure(text=text.upper(), fg=colour),
        )

    def _create_action_window(self) -> None:
        window = tk.Toplevel(self.root)
        self._action_window = window
        window.title("Procedure Action Control")
        window.geometry("470x285")
        window.minsize(430, 265)
        window.resizable(False, False)
        window.configure(bg="#151a21")
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", window.iconify)

        header = tk.Frame(window, bg="#202733", padx=22, pady=16)
        header.pack(fill="x")
        self.action_status_label = tk.Label(
            header,
            text="PREPARING FIRST ACTION",
            bg="#202733",
            fg="#7ecfff",
            font=("Arial", 12, "bold"),
        )
        self.action_status_label.pack(anchor="w")
        self.action_progress_label = tk.Label(
            header,
            text="Stage 0 of 4",
            bg="#202733",
            fg="#8f9bad",
            font=("Arial", 9),
        )
        self.action_progress_label.pack(anchor="w", pady=(4, 0))

        content = tk.Frame(window, bg="#151a21", padx=22, pady=18)
        content.pack(expand=True, fill="both")
        self.action_title_label = tk.Label(
            content,
            text="Waiting for the procedure to initialise",
            bg="#151a21",
            fg="white",
            font=("Arial", 13, "bold"),
            anchor="w",
        )
        self.action_title_label.pack(fill="x")
        self.action_detail_label = tk.Label(
            content,
            text=(
                "The first action will appear here. Nothing executes until "
                "you select Next; Skip step advances without execution."
            ),
            bg="#151a21",
            fg="#aeb8c8",
            font=("Arial", 10),
            justify="left",
            anchor="w",
            wraplength=420,
        )
        self.action_detail_label.pack(fill="x", pady=(8, 18))
        button_row = tk.Frame(content, bg="#151a21")
        button_row.pack(fill="x")
        self.skip_button = tk.Button(
            button_row,
            text="SKIP STEP",
            command=self._skip_next_stage,
            state="disabled",
            disabledforeground="#6f7783",
            bg="#2a3442",
            fg="white",
            activebackground="#8a5a18",
            activeforeground="white",
            relief="flat",
            font=("Arial", 10, "bold"),
            padx=20,
            pady=10,
            cursor="hand2",
        )
        self.skip_button.pack(side="left")
        self.next_button = tk.Button(
            button_row,
            text="NEXT  →",
            command=self._release_next_stage,
            state="disabled",
            disabledforeground="#6f7783",
            bg="#2a3442",
            fg="white",
            activebackground="#287fd1",
            activeforeground="white",
            relief="flat",
            font=("Arial", 11, "bold"),
            padx=30,
            pady=10,
            cursor="hand2",
        )
        self.next_button.pack(side="right")

        window.lift()
        window.attributes("-topmost", True)
        window.after(1200, lambda: window.attributes("-topmost", False))

    def request_next_stage(
        self,
        stage_number: int,
        title: str,
        detail: str,
        event: threading.Event,
    ) -> None:
        self._next_event = event
        self._pending_action_title = title
        self._pending_action_mode = "start"
        self._stage_decision = None

        def update() -> None:
            if self._action_window is None or not self._action_window.winfo_exists():
                self._create_action_window()
            self.action_status_label.configure(
                text="READY FOR NEXT ACTION",
                fg="#66cc88",
            )
            self.action_progress_label.configure(text=f"Stage {stage_number} of 4")
            self.action_title_label.configure(text=title)
            self.action_detail_label.configure(
                text=(
                    f"{detail}\nSelect Next to execute, or Skip step to advance "
                    "without sending a command."
                )
            )
            self.next_button.configure(
                state="normal",
                text="NEXT  →",
                bg="#1E90FF",
                activebackground="#287fd1",
            )
            self.skip_button.configure(
                state="normal",
                text="SKIP STEP",
                bg="#6b4616",
                activebackground="#8a5a18",
            )
            self._action_window.deiconify()
            self._action_window.lift()
            self._action_window.focus_force()

        self.root.after(0, update)

    def request_completion_confirmation(
        self,
        stage_number: int,
        title: str,
        detail: str,
        event: threading.Event,
    ) -> None:
        """Wait for the operator to confirm a UF850 pickup has finished."""
        self._next_event = event
        self._pending_action_title = title
        self._pending_action_mode = "confirm"
        self._stage_decision = None

        def update() -> None:
            if self._action_window is None or not self._action_window.winfo_exists():
                self._create_action_window()
            self.action_status_label.configure(
                text="AWAITING COMPLETION CONFIRMATION",
                fg="#7ecfff",
            )
            self.action_progress_label.configure(text=f"Stage {stage_number} of 4")
            self.action_title_label.configure(text=title)
            self.action_detail_label.configure(text=detail)
            self.next_button.configure(
                state="normal",
                text="CONFIRM COMPLETE",
                bg="#1E90FF",
                activebackground="#287fd1",
            )
            self.skip_button.configure(
                state="disabled",
                text="SKIP UNAVAILABLE",
                bg="#2a3442",
            )
            self._action_window.deiconify()
            self._action_window.lift()
            self._action_window.focus_force()

        self.root.after(0, update)

    def request_action_retry(
        self,
        stage_number: int,
        title: str,
        detail: str,
        event: threading.Event,
    ) -> None:
        """Offer an explicit retry without advancing the procedure stage."""
        self._next_event = event
        self._pending_action_title = title
        self._pending_action_mode = "retry"
        self._stage_decision = None

        def update() -> None:
            if self._action_window is None or not self._action_window.winfo_exists():
                self._create_action_window()
            self.action_status_label.configure(
                text="ACTION NEEDS ATTENTION",
                fg="#ff6666",
            )
            self.action_progress_label.configure(text=f"Stage {stage_number} of 4")
            self.action_title_label.configure(text=title)
            self.action_detail_label.configure(text=detail)
            self.next_button.configure(
                state="normal",
                text="RETRY ACTION",
                bg="#b66a00",
                activebackground="#d98200",
            )
            self.skip_button.configure(
                state="disabled",
                text="SKIP UNAVAILABLE",
                bg="#2a3442",
            )
            self._action_window.deiconify()
            self._action_window.lift()
            self._action_window.focus_force()

        self.root.after(0, update)

    def _release_next_stage(self) -> None:
        event = self._next_event
        if event is None:
            return
        self._next_event = None
        self._stage_decision = "execute"
        if self._pending_action_mode == "confirm":
            self.action_status_label.configure(text="COMPLETION CONFIRMED", fg="#66cc88")
            self.action_detail_label.configure(
                text=(
                    f"{self._pending_action_title} has been confirmed complete.\n"
                    "The agent can now evaluate the result and continue."
                )
            )
            button_text = "CONFIRMED"
        elif self._pending_action_mode == "retry":
            self.action_status_label.configure(text="RETRY IN PROGRESS", fg="#FFA500")
            self.action_detail_label.configure(
                text=(
                    f"{self._pending_action_title} is being reacquired and retried.\n"
                    "Controls remain locked while the agent evaluates the new attempt."
                )
            )
            button_text = "RETRYING"
        else:
            self.action_status_label.configure(text="ACTION IN PROGRESS", fg="#FFA500")
            self.action_detail_label.configure(
                text=(
                    f"{self._pending_action_title} is now executing.\n"
                    "Controls remain locked until the agent reports the next state."
                )
            )
            button_text = "IN PROGRESS"
        self.next_button.configure(
            state="disabled",
            text=button_text,
            bg="#2a3442",
        )
        self.skip_button.configure(
            state="disabled",
            text="SKIP STEP",
            bg="#2a3442",
        )
        self._pending_action_mode = "start"
        event.set()

    def _skip_next_stage(self) -> None:
        """Advance one ready stage without dispatching its action."""
        event = self._next_event
        if event is None or self._pending_action_mode != "start":
            return
        self._next_event = None
        self._stage_decision = "skip"
        self.action_status_label.configure(text="STEP SKIPPED", fg="#ffb347")
        self.action_detail_label.configure(
            text=(
                f"{self._pending_action_title} was skipped.\n"
                "No robot or tool command was sent for this stage."
            )
        )
        self.next_button.configure(
            state="disabled",
            text="SKIPPED",
            bg="#2a3442",
        )
        self.skip_button.configure(
            state="disabled",
            text="SKIPPED",
            bg="#2a3442",
        )
        self._pending_action_mode = "start"
        event.set()

    @property
    def stage_decision(self) -> str | None:
        return self._stage_decision

    def set_action_attention(self, message: str) -> None:
        def update() -> None:
            self.action_status_label.configure(text="ACTION NEEDS ATTENTION", fg="#ff6666")
            self.action_detail_label.configure(text=message)
            self.next_button.configure(
                state="disabled",
                text="NEXT LOCKED",
                bg="#2a3442",
            )
            self.skip_button.configure(
                state="disabled",
                text="SKIP UNAVAILABLE",
                bg="#2a3442",
            )
            if self._action_window is not None:
                self._action_window.deiconify()
                self._action_window.lift()

        self.root.after(0, update)

    def mark_finished(self, skipped_stages: list[str] | None = None) -> None:
        skipped_stages = list(skipped_stages or [])

        def update() -> None:
            if skipped_stages:
                self.action_status_label.configure(
                    text="SEQUENCE COMPLETE — STEPS SKIPPED",
                    fg="#ffb347",
                )
                self.action_progress_label.configure(text="4 of 4 stages resolved")
                self.action_title_label.configure(text="Laptop sequence finished")
                self.action_detail_label.configure(
                    text="Skipped: " + ", ".join(skipped_stages)
                )
            else:
                self.action_status_label.configure(text="PROCEDURE COMPLETE", fg="#66cc66")
                self.action_progress_label.configure(text="4 of 4 stages complete")
                self.action_title_label.configure(text="Laptop disassembly complete")
                self.action_detail_label.configure(
                    text="The requested heat-sink and battery actions have finished."
                )
            self.next_button.configure(
                state="disabled",
                text="COMPLETE",
                bg="#1a3a1a",
            )
            self.skip_button.configure(
                state="disabled",
                text="SKIP STEP",
                bg="#2a3442",
            )

        self.root.after(0, update)

    def _close(self) -> None:
        self._on_stop()
        self.root.destroy()

    def mainloop(self) -> None:
        self.root.mainloop()


class LaptopAutoAgent(Node):
    """Runs the fixed laptop sequence while presenting ReAct-style stages."""

    def __init__(self, config_path: Path):
        super().__init__("laptop_auto_agent")
        self.config_path = Path(config_path)
        self.device_config = DeviceConfig.load(self.config_path)
        # This process contains two ROS nodes.  Ignore the launch-level
        # ``__node:=laptop_auto_agent`` remap for the embedded skill so it keeps
        # its own graph name and rosout publisher.
        self.unscrew_skill = UnscrewSkill(
            device_cfg=self.device_config,
            use_global_arguments=False,
            prefer_remote_exotica=True,
        )
        self.dashboard: ReActDashboard | None = None
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._skipped_stages: list[str] = []

        self.declare_parameter("vision_timeout_s", 120.0)
        self.declare_parameter("stage_delay_s", 0.65)

    @property
    def vision_timeout_s(self) -> float:
        return float(self.get_parameter("vision_timeout_s").value)

    @property
    def stage_delay_s(self) -> float:
        return max(float(self.get_parameter("stage_delay_s").value), 0.0)

    def attach_dashboard(self, dashboard: ReActDashboard) -> None:
        self.dashboard = dashboard

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run_sequence,
            name="laptop-auto-sequence",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.unscrew_skill.tool_pub.publish(Int8(data=0))
            self.unscrew_skill.moveit_backend.stop_servo(timeout_sec=2.0)
            self.unscrew_skill._pub_state("IDLE")
        except Exception as exc:
            self.get_logger().warning(f"Could not complete xArm5 stop request: {exc}")
        if self.dashboard:
            self.dashboard.set_status("Execution stopped", "#ff6666")
            self.dashboard.set_action_attention(
                "Execution was stopped. No further action can be released."
            )

    def _emit(self, node_name: str, text: str, *, delay: bool = True) -> None:
        self.get_logger().info(f"[{node_name}] {text}")
        if self.dashboard:
            self.dashboard.emit(node_name, text)
        if delay and self.stage_delay_s:
            self._stop_event.wait(self.stage_delay_s)

    @staticmethod
    def _valid_target(target: dict) -> bool:
        xyz = target.get("xyz")
        return (
            target.get("id") is not None
            and isinstance(xyz, (list, tuple))
            and len(xyz) >= 3
            and all(value is not None for value in xyz[:3])
        )

    def _find_exact_target(self, target_label: str) -> dict | None:
        target_label = target_label.strip().lower()
        with self.unscrew_skill.data_lock:
            targets = copy.deepcopy(self.unscrew_skill.latest_targets)
        return next(
            (
                target
                for target in targets
                if str(target.get("label", "")).strip().lower() == target_label
                and self._valid_target(target)
            ),
            None,
        )

    def _wait_for_target(self, target_label: str) -> dict | None:
        deadline = time.monotonic() + self.vision_timeout_s
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            target = self._find_exact_target(target_label)
            if target is not None:
                return target
            self._stop_event.wait(0.2)
        return None

    def _apply_laptop_unscrew_parameters(self, target_label: str) -> None:
        # battery_screw is currently a detector label rather than a sequence
        # step in laptop.yaml. Both targets use the calibrated laptop screw
        # settings until a separate battery calibration is added.
        configured_labels = {
            step.target
            for step in self.device_config.disassembly_sequence
            if step.action == "unscrew"
        }
        config_label = target_label if target_label in configured_labels else "heat_sink_screw"
        self.unscrew_skill._apply_unscrew_config(
            self.device_config,
            target_label=config_label,
        )

    def _wait_for_next(
        self,
        stage_number: int,
        title: str,
        detail: str,
    ) -> str | None:
        """Block a procedure stage until its explicit Next release."""
        if self.dashboard is None:
            self.get_logger().error(
                f"Cannot release stage {stage_number} without the action-control window"
            )
            return False

        event = threading.Event()
        self.dashboard.set_status(f"Awaiting Next: {title}", "#7ecfff")
        self.dashboard.request_next_stage(stage_number, title, detail, event)
        self.get_logger().info(
            f"[GATE] Stage {stage_number} ready: {title}. Select NEXT or SKIP STEP."
        )
        while not self._stop_event.is_set():
            if event.wait(timeout=0.2):
                decision = self.dashboard.stage_decision or "execute"
                self.get_logger().info(
                    f"[GATE] Stage {stage_number} decision: {decision}."
                )
                return decision
        return None

    def _record_skipped_stage(self, stage_number: int, title: str) -> None:
        self._skipped_stages.append(title)
        self._emit(
            "ACT",
            f"Observation: Stage {stage_number} ({title}) was skipped by the operator.\n"
            "No robot or tool command was sent; the sequence advanced to the next stage.",
        )

    def _wait_for_pickup_confirmation(
        self,
        stage_number: int,
        component: str,
    ) -> bool:
        """Block until the operator confirms the active UF850 pickup is done."""
        if self.dashboard is None:
            self.get_logger().error(
                f"Cannot confirm stage {stage_number} without the action-control window"
            )
            return False

        event = threading.Event()
        title = f"Pick up {component}"
        self.dashboard.set_status(
            f"Awaiting completion confirmation: {component}",
            "#7ecfff",
        )
        self.dashboard.request_completion_confirmation(
            stage_number,
            title,
            (
                f"The UF850 pickup policy for the {component} is active.\n"
                "When the component is clear of the laptop, select Confirm complete."
            ),
            event,
        )
        while not self._stop_event.is_set():
            if event.wait(timeout=0.2):
                return True
        return False

    def _wait_for_unscrew_retry(
        self,
        stage_number: int,
        target: UnscrewTarget,
    ) -> bool:
        """Wait for an explicit retry request after a failed unscrew attempt."""
        if self.dashboard is None:
            self.get_logger().error(
                f"Cannot retry stage {stage_number} without the action-control window"
            )
            return False

        event = threading.Event()
        title = f"Remove {target.display_name}"
        self.dashboard.set_status(f"Retry available: {target.display_name}", "#ff6666")
        self.dashboard.request_action_retry(
            stage_number,
            title,
            (
                f"The {target.display_name} action did not complete. Inspect the target, "
                "tool-camera view, and robot clearance.\n"
                "Select Retry action to reacquire the target and repeat this stage."
            ),
            event,
        )
        while not self._stop_event.is_set():
            if event.wait(timeout=0.2):
                self._emit(
                    "THINK",
                    f"Reasoning: A retry was requested for the {target.display_name}.\n"
                    "The agent will reacquire fresh vision data and repeat the same stage without advancing.",
                )
                return True
        return False

    def _run_unscrew(self, target: UnscrewTarget) -> bool:
        if self.dashboard:
            self.dashboard.set_status(f"Locating {target.display_name}")
        self._emit(
            "VISION",
            f"Scene update received. Searching for {target.label}.\n"
            "Validating the target ID and its 3D position before execution.",
        )
        detection = self._wait_for_target(target.label)
        if detection is None:
            if self.dashboard:
                self.dashboard.set_status("Needs attention", "#ff6666")
                self.dashboard.set_action_attention(
                    f"No valid {target.label} target was available. "
                    "The procedure cannot advance."
                )
            self._emit(
                "ACT",
                f"Observation: No valid {target.label} target was detected before the timeout.\n"
                "Execution is paused because a verified 3D target is required.",
                delay=False,
            )
            return False

        target_id = detection.get("id")
        self._emit(
            "THINK",
            f"Reasoning: The {target.display_name} must be removed before "
            f"the {target.pickup_name} can be extracted.\n"
            "The target has a valid 3D pose and the xArm5 tool arm is ready.",
        )
        self._emit(
            "PLAN",
            f"Plan: Align xArm5 with the detected {target.label}.\n"
            "Engage the screwdriver, remove the screw, and return to a safe pose.",
        )
        self._emit(
            "ACTION",
            f'Action: unscrew(unscrew_id={target_id}, unscrew_label="{target.label}")\n'
            f"Command route: xArm5 unscrew policy → {target.label}.",
        )

        if self._stop_event.is_set():
            return False
        if self.dashboard:
            self.dashboard.set_status("xArm5 unscrewing", "#FFA500")
        self._apply_laptop_unscrew_parameters(target.label)
        result = self.unscrew_skill.execute_unscrew_command(
            target_id=target_id,
            target_label=target.label,
            interactive=False,
            target_data_override=detection,
        )
        if result is True:
            self._emit(
                "ACT",
                f"Observation: {target.display_name.capitalize()} removed successfully.\n"
                "The xArm5 tool arm completed the operation and returned to its safe pose.",
            )
            return True

        if result == "HOLE":
            self._emit(
                "ACT",
                f"Observation: The tool camera classified the {target.label} position as an open hole, "
                "so screwdriver engagement did not run.\n"
                "Removal is unverified; execution is paused instead of treating this as a successful unscrew.",
                delay=False,
            )
            if self.dashboard:
                self.dashboard.set_status("Needs attention", "#ff6666")
                self.dashboard.set_action_attention(
                    f"The tool camera saw a hole instead of a stable {target.label}. "
                    "The screw was not engaged. Inspect the target or vision result, then restart the procedure."
                )
            return False

        self._emit(
            "ACT",
            f"Observation: The {target.display_name} operation did not complete. "
            "Execution has stopped for inspection.\n"
            "The next ReAct stage will remain locked until this condition is resolved.",
            delay=False,
        )
        if self.dashboard:
            self.dashboard.set_status("Needs attention", "#ff6666")
            self.dashboard.set_action_attention(
                f"The {target.display_name} action did not complete. "
                "Inspect the robot and tool before restarting the procedure."
            )
        return False

    def _run_pickup_policy(self, component: str, stage_number: int) -> bool:
        self._emit(
            "THINK",
            f"Reasoning: The {component} is now ready for extraction.\n"
            "The preceding screw-removal dependency has been completed successfully.",
        )
        self._emit(
            "PLAN",
            f"Plan: Execute the UF850 pickup policy for the {component}.\n"
            "Maintain a controlled grasp and lift the component clear of the laptop.",
        )
        self._emit(
            "ACTION",
            f'Action: pickup_policy(arm="uf850", target="{component.replace(" ", "_")}")\n'
            f"Command route: UF850 pickup policy → {component}.",
        )
        if self.dashboard:
            self.dashboard.set_status(
                f"Executing UF850 {component} pickup policy",
                "#1E90FF",
            )

        if self._stop_event.is_set():
            return False
        if not self._wait_for_pickup_confirmation(stage_number, component):
            return False
        self._emit(
            "ACT",
            f"Observation: {component.capitalize()} pickup policy completed successfully.\n"
            "Completion was confirmed and the following dependency is now cleared.",
        )
        return True

    def _run_sequence(self) -> None:
        try:
            if self.dashboard:
                self.dashboard.set_status("Autonomous execution active", "#66cc66")
            self._emit(
                "VISION",
                "Mission received: Disassemble the laptop heat sink and battery.\n"
                "The agent will coordinate xArm5 unscrewing and UF850 pickup policies.",
            )

            stage_number = 1
            for target in AUTOMATIC_SEQUENCE:
                unscrew_title = f"Remove {target.display_name}"
                unscrew_decision = self._wait_for_next(
                    stage_number,
                    unscrew_title,
                    f"xArm5 will run the unscrew policy for {target.label}.",
                )
                if unscrew_decision is None:
                    return
                if unscrew_decision == "skip":
                    self._record_skipped_stage(stage_number, unscrew_title)
                else:
                    while not self._stop_event.is_set():
                        if self._run_unscrew(target):
                            break
                        if not self._wait_for_unscrew_retry(stage_number, target):
                            return
                    if self._stop_event.is_set():
                        return
                stage_number += 1

                pickup_title = f"Pick up {target.pickup_name}"
                pickup_decision = self._wait_for_next(
                    stage_number,
                    pickup_title,
                    f"UF850 will run the pickup policy for the {target.pickup_name}.",
                )
                if pickup_decision is None:
                    return
                if pickup_decision == "skip":
                    self._record_skipped_stage(stage_number, pickup_title)
                elif not self._run_pickup_policy(target.pickup_name, stage_number):
                    return
                stage_number += 1

            if self._skipped_stages:
                skipped = ", ".join(self._skipped_stages)
                self._emit(
                    "THINK",
                    "Reasoning: Every procedure stage has now been executed or explicitly skipped.\n"
                    f"Operator-skipped stages: {skipped}.",
                )
                self._emit(
                    "PLAN",
                    "Final Answer: Laptop procedure sequence finished with skipped stages.\n"
                    f"Skipped without robot execution: {skipped}.",
                    delay=False,
                )
            else:
                self._emit(
                    "THINK",
                    "Reasoning: The heat sink and battery have been removed successfully. "
                    "All requested disassembly operations are complete.\n"
                    "No remaining action exists in the requested laptop procedure.",
                )
                self._emit(
                    "PLAN",
                    "Final Answer: Laptop disassembly completed successfully.\n"
                    "Both removal stages have reported successful completion.",
                    delay=False,
                )
            if self.dashboard:
                if self._skipped_stages:
                    self.dashboard.set_status("Sequence complete with skipped stages", "#ffb347")
                else:
                    self.dashboard.set_status("Procedure complete", "#66cc66")
                self.dashboard.mark_finished(self._skipped_stages)
        except Exception as exc:
            self.get_logger().exception("Laptop sequence failed")
            self._emit(
                "ACT",
                f"Observation: Execution stopped because of {type(exc).__name__}: {exc}\n"
                "The next ReAct stage is locked until the failure is resolved.",
                delay=False,
            )
            if self.dashboard:
                self.dashboard.set_status("Needs attention", "#ff6666")
                self.dashboard.set_action_attention(
                    "The current action stopped unexpectedly. Review the dashboard log "
                    "before restarting the procedure."
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    agent = LaptopAutoAgent(DEFAULT_CONFIG)
    dashboard = ReActDashboard(agent.stop)
    agent.attach_dashboard(dashboard)

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(agent)
    executor.add_node(agent.unscrew_skill)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Begin automatically once the UI and ROS callbacks are active.
    dashboard.root.after(1000, agent.start)
    try:
        dashboard.mainloop()
    finally:
        agent.stop()
        executor.shutdown()
        agent.unscrew_skill.destroy_node()
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
