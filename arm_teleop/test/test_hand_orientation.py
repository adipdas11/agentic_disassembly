import math

import numpy as np
import pytest

from arm_teleop.hand_math import (
    DEFAULT_HAND_TO_TOOL_ROTATION,
    apply_translation_deadzone,
    matrix_to_quat,
    palm_frame_matrix,
    palm_frame_quaternion,
    pose_pivot_position,
    pose_position_from_pivot,
    quat_conjugate,
    quat_multiply,
    quat_rotate_vector,
    quat_to_matrix,
    quat_to_rotation_vector,
    quat_to_rpy,
    retarget_hand_orientation,
    rotation_vector_to_quat,
    rpy_to_quat,
    validate_rotation_matrix,
)
from arm_teleop.vuer_hand_tracking import canonical_palm_rotation_from_webxr_wrist


def _palm_landmarks(handedness="right"):
    wrist = np.array([0.0, 0.0, 0.0])
    middle = np.array([1.0, 0.0, 0.0])
    if handedness == "right":
        index = np.array([0.85, 0.45, 0.0])
        pinky = np.array([0.85, -0.45, 0.0])
    else:
        index = np.array([0.85, -0.45, 0.0])
        pinky = np.array([0.85, 0.45, 0.0])
    return wrist, index, middle, pinky


def _rotate_landmarks(landmarks, quaternion, basis=None):
    rotation = np.asarray(quat_to_matrix(quaternion))
    source_basis = np.eye(3) if basis is None else np.asarray(basis)
    return tuple(source_basis @ rotation @ np.asarray(point) for point in landmarks)


def _assert_same_rotation(actual, expected, atol=1e-7):
    np.testing.assert_allclose(quat_to_matrix(actual), quat_to_matrix(expected), atol=atol)


def test_translation_deadzone_rejects_orientation_tracking_jitter():
    shaped = apply_translation_deadzone(
        [0.020, -0.024, 0.0],
        [0.025, 0.025, 0.025],
    )

    np.testing.assert_allclose(shaped, [0.0, 0.0, 0.0])


def test_translation_deadzone_is_continuous_and_preserves_direction():
    shaped = apply_translation_deadzone(
        [0.040, -0.055, 0.025],
        [0.025, 0.025, 0.025],
    )

    np.testing.assert_allclose(shaped, [0.015, -0.030, 0.0])


def test_translation_deadzone_rejects_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        apply_translation_deadzone([0.0, 0.0, 0.0], [0.0, -0.01, 0.0])


def test_quaternion_vector_rotation_preserves_metric_length():
    vector = [0.0, 0.0, 0.28]

    rotated = quat_rotate_vector(rpy_to_quat(0.0, math.pi / 2.0, 0.0), vector)

    np.testing.assert_allclose(rotated, [0.28, 0.0, 0.0], atol=1e-7)


def test_tcp_rotation_keeps_robot_wrist_pivot_fixed():
    tcp_origin = [0.8, -0.1, 1.0]
    origin_quat = rpy_to_quat(0.0, 0.0, 0.0)
    target_quat = rpy_to_quat(0.0, math.pi / 3.0, 0.0)
    tcp_to_wrist = [0.0, 0.0, 0.28]
    wrist_pivot = pose_pivot_position(tcp_origin, origin_quat, tcp_to_wrist)

    target_tcp = pose_position_from_pivot(wrist_pivot, target_quat, tcp_to_wrist)

    np.testing.assert_allclose(
        pose_pivot_position(target_tcp, target_quat, tcp_to_wrist),
        wrist_pivot,
        atol=1e-7,
    )
    assert not np.allclose(target_tcp, tcp_origin)


@pytest.mark.parametrize("handedness", ["left", "right"])
def test_palm_frame_is_right_handed_for_both_hands(handedness):
    matrix = palm_frame_matrix(*_palm_landmarks(handedness), handedness)

    assert matrix is not None
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-7)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("delta", "expected_rpy"),
    [
        (rpy_to_quat(0.35, 0.0, 0.0), (0.35, 0.0, 0.0)),
        (rpy_to_quat(0.0, -0.30, 0.0), (0.0, -0.30, 0.0)),
        (rpy_to_quat(0.0, 0.0, 0.25), (0.0, 0.0, 0.25)),
    ],
)
def test_synthetic_wrist_roll_pitch_and_yaw_use_anatomical_axes(delta, expected_rpy):
    landmarks = _palm_landmarks("right")
    neutral = palm_frame_quaternion(*landmarks, "right")
    current = palm_frame_quaternion(*_rotate_landmarks(landmarks, delta), "right")

    assert neutral is not None
    assert current is not None
    local_delta = quat_multiply(quat_conjugate(neutral), current)
    np.testing.assert_allclose(quat_to_rpy(local_delta), expected_rpy, atol=1e-7)


def test_equivalent_vuer_wrist_and_mediapipe_landmarks_have_same_relative_rotation():
    landmarks = _palm_landmarks("right")
    motion = rpy_to_quat(0.20, -0.15, 0.30)

    neutral_points = landmarks
    current_points = _rotate_landmarks(landmarks, motion)
    mediapipe_neutral = palm_frame_quaternion(*neutral_points, "right")
    mediapipe_current = palm_frame_quaternion(*current_points, "right")
    assert mediapipe_neutral is not None
    assert mediapipe_current is not None
    mediapipe_delta = quat_multiply(quat_conjugate(mediapipe_neutral), mediapipe_current)

    wrist_to_palm = np.diag([1.0, -1.0, -1.0])
    neutral_wrist = np.eye(4)
    neutral_wrist[:3, :3] = wrist_to_palm
    current_wrist = np.eye(4)
    current_wrist[:3, :3] = np.asarray(quat_to_matrix(motion)) @ wrist_to_palm
    vuer_neutral = matrix_to_quat(
        canonical_palm_rotation_from_webxr_wrist(neutral_wrist).tolist()
    )
    vuer_current = matrix_to_quat(
        canonical_palm_rotation_from_webxr_wrist(current_wrist).tolist()
    )
    vuer_delta = quat_multiply(quat_conjugate(vuer_neutral), vuer_current)

    _assert_same_rotation(vuer_delta, mediapipe_delta)
    _assert_same_rotation(vuer_delta, motion)


def test_calibration_identity_produces_no_tool_rotation():
    hand_pose = rpy_to_quat(0.4, -0.2, 0.1)
    tool_pose = rpy_to_quat(-0.3, 0.25, 0.6)

    target = retarget_hand_orientation(
        hand_pose,
        hand_pose,
        tool_pose,
        DEFAULT_HAND_TO_TOOL_ROTATION,
        track_orientation=True,
    )

    _assert_same_rotation(target, tool_pose)


def test_default_hand_to_tool_mapping_maps_roll_to_negative_tool_z():
    angle = 0.4

    target = retarget_hand_orientation(
        rpy_to_quat(0.0, 0.0, 0.0),
        rpy_to_quat(angle, 0.0, 0.0),
        rpy_to_quat(0.0, 0.0, 0.0),
        DEFAULT_HAND_TO_TOOL_ROTATION,
        track_orientation=True,
    )

    _assert_same_rotation(target, rpy_to_quat(0.0, 0.0, -angle))


def test_orientation_scale_reduces_final_rotation_not_only_filter_speed():
    angle = 1.0

    target = retarget_hand_orientation(
        rpy_to_quat(0.0, 0.0, 0.0),
        rpy_to_quat(0.0, angle, 0.0),
        rpy_to_quat(0.0, 0.0, 0.0),
        np.eye(3),
        track_pitch=True,
        orientation_scale_xyz=[1.0, 0.5, 1.0],
    )

    _assert_same_rotation(target, rpy_to_quat(0.0, angle * 0.5, 0.0))


def test_orientation_scale_rejects_negative_gain():
    with pytest.raises(ValueError, match="non-negative"):
        retarget_hand_orientation(
            rpy_to_quat(0.0, 0.0, 0.0),
            rpy_to_quat(0.0, 0.2, 0.0),
            rpy_to_quat(0.0, 0.0, 0.0),
            np.eye(3),
            track_pitch=True,
            orientation_scale_xyz=[1.0, -0.5, 1.0],
        )


def test_default_alignment_matches_parallel_jaw_bottle_grasp():
    alignment = np.asarray(DEFAULT_HAND_TO_TOOL_ROTATION).reshape(3, 3)

    # Matrix columns are the hand X/Y/Z axes expressed in TCP coordinates.
    np.testing.assert_allclose(alignment[:, 0], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(alignment[:, 1], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(alignment[:, 2], [0.0, 1.0, 0.0])
    assert np.linalg.det(alignment) == pytest.approx(1.0)


def test_tool_delta_is_post_multiplied_in_calibrated_tcp_frame():
    angle = 0.3
    tool_origin = rpy_to_quat(0.0, 0.0, math.pi / 2.0)

    target = retarget_hand_orientation(
        rpy_to_quat(0.0, 0.0, 0.0),
        rpy_to_quat(0.0, 0.0, angle),
        tool_origin,
        DEFAULT_HAND_TO_TOOL_ROTATION,
        track_orientation=True,
    )
    expected = quat_multiply(tool_origin, rpy_to_quat(0.0, angle, 0.0))
    wrong_order = quat_multiply(rpy_to_quat(0.0, angle, 0.0), tool_origin)

    _assert_same_rotation(target, expected)
    assert not np.allclose(quat_to_matrix(target), quat_to_matrix(wrong_order))


def test_partial_orientation_keeps_disabled_axes_fixed():
    hand_delta = rotation_vector_to_quat([0.2, -0.3, 0.4])
    target = retarget_hand_orientation(
        rpy_to_quat(0.0, 0.0, 0.0),
        hand_delta,
        rpy_to_quat(0.0, 0.0, 0.0),
        DEFAULT_HAND_TO_TOOL_ROTATION,
        track_pitch=True,
    )

    _assert_same_rotation(target, rpy_to_quat(0.3, 0.0, 0.0))


def test_large_combined_pitch_yaw_does_not_create_false_roll():
    angle = math.pi / 2.0
    component = angle / math.sqrt(2.0)
    hand_delta = rotation_vector_to_quat([0.0, component, component])

    target = retarget_hand_orientation(
        rpy_to_quat(0.0, 0.0, 0.0),
        hand_delta,
        rpy_to_quat(0.0, 0.0, 0.0),
        np.eye(3),
        track_pitch=True,
        track_yaw=True,
    )

    _assert_same_rotation(target, hand_delta)
    assert quat_to_rotation_vector(target)[0] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    ("hand_delta", "expected_tool_delta"),
    [
        (rpy_to_quat(0.0, math.pi / 2.0, 0.0), rpy_to_quat(-math.pi / 2.0, 0.0, 0.0)),
        (rpy_to_quat(0.0, 0.0, math.pi / 2.0), rpy_to_quat(0.0, math.pi / 2.0, 0.0)),
    ],
)
def test_bottle_grasp_pitch_and_yaw_map_to_up_down_and_left_right_tool_axes(
    hand_delta,
    expected_tool_delta,
):
    target = retarget_hand_orientation(
        rpy_to_quat(0.0, 0.0, 0.0),
        hand_delta,
        rpy_to_quat(0.0, 0.0, 0.0),
        DEFAULT_HAND_TO_TOOL_ROTATION,
        track_pitch=True,
        track_yaw=True,
    )

    _assert_same_rotation(target, expected_tool_delta)


@pytest.mark.parametrize(
    "landmarks",
    [
        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.8, 1e-8, 0.0], [1.0, 0.0, 0.0], [0.8, -1e-8, 0.0]),
        ([0.0, 0.0, 0.0], [0.8, math.nan, 0.0], [1.0, 0.0, 0.0], [0.8, -0.4, 0.0]),
    ],
)
def test_degenerate_or_invalid_palm_frames_are_rejected(landmarks):
    assert palm_frame_quaternion(*landmarks, "right") is None


def test_hand_to_tool_rotation_rejects_reflection():
    with pytest.raises(ValueError, match="right-handed"):
        validate_rotation_matrix(np.diag([1.0, 1.0, -1.0]), "test_rotation")
