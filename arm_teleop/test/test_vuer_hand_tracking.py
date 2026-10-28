from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
import yaml

from arm_teleop.vuer_hand_tracking import (
    INDEX_METACARPAL,
    INDEX_TIP,
    PINKY_METACARPAL,
    THUMB_TIP,
    WEBXR_TO_ROS,
    canonical_palm_rotation_from_webxr_wrist,
    gripper_aperture,
    is_vuer_hand_absent,
    parse_vuer_hand,
    update_hysteresis,
    webxr_to_ros_matrices,
)


def _serialized_identity_hand():
    matrices = np.repeat(np.eye(4)[None, :, :], 25, axis=0)
    return matrices.transpose(0, 2, 1).reshape(-1), matrices


def test_parse_vuer_hand_decodes_each_column_major_matrix():
    values, expected = _serialized_identity_hand()
    expected[0, :3, 3] = [1.0, 2.0, 3.0]
    values = expected.transpose(0, 2, 1).reshape(-1)

    parsed = parse_vuer_hand(values)

    np.testing.assert_allclose(parsed, expected)


def test_parse_vuer_hand_rejects_wrong_length():
    with pytest.raises(ValueError, match="Expected 400"):
        parse_vuer_hand([0.0] * 399)


def test_parse_vuer_hand_decodes_packed_float32_webxr_payload():
    values, expected = _serialized_identity_hand()
    expected[0, :3, 3] = [1.0, 2.0, 3.0]
    packed = expected.astype(np.float32).transpose(0, 2, 1).reshape(-1).tobytes()

    parsed = parse_vuer_hand(packed)

    np.testing.assert_allclose(parsed, expected)


@pytest.mark.parametrize(
    "sentinel",
    [
        b"\x00",
        [b"\x00"],
        namedtuple("ExtType", ("code", "data"))(0, b"\x00"),
    ],
)
def test_vuer_empty_binary_hand_sentinel_is_treated_as_absent(sentinel):
    assert is_vuer_hand_absent(sentinel)


def test_vuer_nonempty_extension_payload_is_not_treated_as_absent():
    extension = namedtuple("ExtType", ("code", "data"))(0, b"not-an-empty-hand")

    assert not is_vuer_hand_absent(extension)


def test_packed_vuer_hand_is_not_treated_as_absent():
    values, _ = _serialized_identity_hand()
    assert not is_vuer_hand_absent(values.astype(np.float32).tobytes())


def test_webxr_to_ros_maps_right_up_backward_to_forward_left_up():
    _, matrices = _serialized_identity_hand()
    matrices[0, :3, 3] = [1.0, 2.0, 3.0]

    converted = webxr_to_ros_matrices(matrices)

    np.testing.assert_allclose(converted[0, :3, 3], [-3.0, -1.0, 2.0])
    np.testing.assert_allclose(converted[0, :3, :3], np.eye(3))


def test_quest_webxr_forward_maps_to_positive_robot_x():
    _, matrices = _serialized_identity_hand()
    matrices[0, :3, 3] = [0.0, 0.0, -0.20]
    ros_forward = webxr_to_ros_matrices(matrices)[0, :3, 3]
    config_path = Path(__file__).parents[1] / "config" / "quest_exotica_teleop.yaml"
    parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "exotica_arm_teleop"
    ]["ros__parameters"]
    translation_mapping = np.asarray(parameters["camera_to_base_rotation"]).reshape(3, 3)

    robot_delta = translation_mapping @ ros_forward

    np.testing.assert_allclose(ros_forward, [0.20, 0.0, 0.0])
    np.testing.assert_allclose(robot_delta, [0.20, 0.0, 0.0])


def test_webxr_wrist_axes_map_to_canonical_palm_axes():
    _, matrices = _serialized_identity_hand()
    converted = webxr_to_ros_matrices(matrices)

    palm_rotation = canonical_palm_rotation_from_webxr_wrist(converted[0])

    np.testing.assert_allclose(palm_rotation, np.diag([1.0, -1.0, -1.0]))
    assert np.linalg.det(palm_rotation) == pytest.approx(1.0)


def test_rotated_webxr_wrist_maps_native_longitudinal_lateral_and_normal_axes():
    _, matrices = _serialized_identity_hand()
    angle = 0.63
    raw_wrist_rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    matrices[0, :3, :3] = raw_wrist_rotation

    converted = webxr_to_ros_matrices(matrices)
    actual = canonical_palm_rotation_from_webxr_wrist(converted[0])

    # WebXR wrist axes: canonical X=-Z, Y=+X, Z=-Y.
    native_wrist_to_palm = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    expected = WEBXR_TO_ROS @ raw_wrist_rotation @ native_wrist_to_palm
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_webxr_wrist_orientation_is_independent_of_landmark_position_noise():
    _, matrices = _serialized_identity_hand()
    converted = webxr_to_ros_matrices(matrices)
    expected = canonical_palm_rotation_from_webxr_wrist(converted[0])

    converted[INDEX_METACARPAL, :3, 3] = [8.0, -3.0, 2.0]
    converted[PINKY_METACARPAL, :3, 3] = [-5.0, 7.0, -1.0]
    actual = canonical_palm_rotation_from_webxr_wrist(converted[0])

    np.testing.assert_allclose(actual, expected)


def test_webxr_wrist_translation_does_not_change_palm_orientation():
    _, matrices = _serialized_identity_hand()
    angle = 0.41
    matrices[0, :3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = canonical_palm_rotation_from_webxr_wrist(webxr_to_ros_matrices(matrices)[0])

    translated = matrices.copy()
    translated[:, :3, 3] += np.array([2.0, -3.0, 4.0])
    actual = canonical_palm_rotation_from_webxr_wrist(webxr_to_ros_matrices(translated)[0])

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_webxr_wrist_orientation_rejects_invalid_rotation():
    wrist = np.eye(4)
    wrist[0, 0] = 2.0

    with pytest.raises(ValueError, match="orthonormal"):
        canonical_palm_rotation_from_webxr_wrist(wrist)


def test_gripper_aperture_is_normalized_by_palm_width():
    _, matrices = _serialized_identity_hand()
    matrices[INDEX_METACARPAL, :3, 3] = [0.0, 0.0, 0.0]
    matrices[PINKY_METACARPAL, :3, 3] = [0.0, 0.08, 0.0]
    matrices[THUMB_TIP, :3, 3] = [0.0, 0.0, 0.0]
    matrices[INDEX_TIP, :3, 3] = [0.0, 0.08, 0.0]

    aperture, ratio = gripper_aperture(matrices, minimum_ratio=0.0, maximum_ratio=2.0)

    assert ratio == pytest.approx(1.0)
    assert aperture == pytest.approx(0.5)


def test_distance_hysteresis_uses_separate_on_and_off_thresholds():
    assert update_hysteresis(False, 0.02, 0.025, 0.035)
    assert update_hysteresis(True, 0.03, 0.025, 0.035)
    assert not update_hysteresis(True, 0.04, 0.025, 0.035)
