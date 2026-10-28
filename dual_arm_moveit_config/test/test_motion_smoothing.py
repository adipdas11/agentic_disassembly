import math

from dual_arm_moveit_config.motion_backend import MotionBackend, quintic_smoothstep


def test_quintic_smoothstep_has_safe_endpoints_and_is_monotonic():
    samples = [quintic_smoothstep(index / 100.0) for index in range(101)]

    assert samples[0] == 0.0
    assert samples[-1] == 1.0
    assert all(right >= left for left, right in zip(samples, samples[1:]))


def test_quintic_smoothstep_clamps_out_of_range_input():
    assert quintic_smoothstep(-1.0) == 0.0
    assert quintic_smoothstep(2.0) == 1.0


def test_quintic_sampling_honours_joint_step_bound():
    max_delta = 1.2
    requested_step = 0.002
    steps = math.ceil((1.875 * max_delta) / requested_step)
    samples = [
        max_delta * quintic_smoothstep(index / steps)
        for index in range(steps + 1)
    ]

    assert max(right - left for left, right in zip(samples, samples[1:])) <= (
        requested_step + 1e-9
    )


def test_rg6_force_is_clamped_to_hardware_maximum():
    class Publisher:
        def __init__(self):
            self.message = None

        def publish(self, message):
            self.message = message

    backend = MotionBackend.__new__(MotionBackend)
    backend._gripper_force_pub = Publisher()
    backend.current_gripper_force_n = 0.0

    assert backend.set_gripper_force(999.0)
    assert backend.current_gripper_force_n == 120.0
    assert backend._gripper_force_pub.message.data == 120.0
