"""Pure helpers for converting Vuer WebXR hand data into ROS conventions."""

from __future__ import annotations

import numpy as np


JOINT_NAMES = (
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
)

WRIST = 0
THUMB_TIP = 4
INDEX_METACARPAL = 5
INDEX_TIP = 9
MIDDLE_METACARPAL = 10
PINKY_METACARPAL = 20
PINKY_TIP = 24

# WebXR/Three.js: +X right, +Y up, -Z forward.
# ROS teleop frame: +X forward, +Y left, +Z up.
WEBXR_TO_ROS = np.asarray(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# WebXR defines wrist -Z toward the palm and wrist -Y out through the palm.
# Therefore the canonical palm axes expressed in the native wrist frame are
# X=-Z, Y=+X, Z=-Y. ``webxr_to_ros_matrices`` changes both the reference and
# local coordinate bases with B*R*B^T, so the corresponding post-multiplier in
# that converted basis is diag(1, -1, -1). The result's columns are the shared
# longitudinal X, lateral Y, and outward-normal Z axes in ROS coordinates.
ROS_WEBXR_WRIST_TO_CANONICAL_PALM = np.diag([1.0, -1.0, -1.0])


def is_vuer_hand_absent(values) -> bool:
    """Return whether Vuer sent its empty binary sentinel for an untracked hand."""

    if isinstance(values, (bytes, bytearray, memoryview)):
        return memoryview(values).nbytes <= 1
    # Vuer serializes an absent hand as a MessagePack extension with code 0 and
    # data=b"\x00". Depending on the decoder, this reaches us as an ExtType-like
    # tuple rather than as the raw byte payload. Keep this duck-typed so the
    # pure conversion helpers do not require msgpack at runtime.
    extension_code = getattr(values, "code", None)
    extension_data = getattr(values, "data", None)
    if extension_code == 0 and isinstance(extension_data, (bytes, bytearray, memoryview)):
        return memoryview(extension_data).nbytes <= 1
    if isinstance(values, (list, tuple)) and len(values) == 1:
        return is_vuer_hand_absent(values[0])
    return False


def parse_vuer_hand(values) -> np.ndarray:
    """Return 25 WebXR matrices as ``(25, 4, 4)`` row-major numpy arrays.

    Vuer concatenates 25 WebGL matrices. Each individual matrix is serialized
    column-major, so every 16-value block must be transposed after reshaping.
    """

    expected = len(JOINT_NAMES) * 16
    if isinstance(values, (bytes, bytearray, memoryview)):
        packed = memoryview(values)
        expected_bytes = expected * np.dtype(np.float32).itemsize
        if packed.nbytes != expected_bytes:
            raise ValueError(
                f"Expected {expected_bytes} packed Vuer hand bytes, received {packed.nbytes}."
            )
        flat = np.frombuffer(packed, dtype="<f4").astype(np.float64)
    else:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size != expected:
        raise ValueError(f"Expected {expected} Vuer hand values, received {flat.size}.")
    if not np.all(np.isfinite(flat)):
        raise ValueError("Vuer hand data contains non-finite values.")
    return flat.reshape(len(JOINT_NAMES), 4, 4).transpose(0, 2, 1)


def webxr_to_ros_matrices(webxr_matrices: np.ndarray) -> np.ndarray:
    """Change the basis of WebXR joint transforms to ROS axes."""

    matrices = np.asarray(webxr_matrices, dtype=np.float64)
    if matrices.shape != (len(JOINT_NAMES), 4, 4):
        raise ValueError(f"Expected hand matrix shape (25, 4, 4), received {matrices.shape}.")

    converted = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(JOINT_NAMES), axis=0)
    converted[:, :3, :3] = WEBXR_TO_ROS @ matrices[:, :3, :3] @ WEBXR_TO_ROS.T
    converted[:, :3, 3] = (WEBXR_TO_ROS @ matrices[:, :3, 3].T).T
    return converted


def canonical_palm_rotation_from_webxr_wrist(ros_wrist_matrix: np.ndarray) -> np.ndarray:
    """Return the canonical palm rotation from a ROS-basis WebXR wrist pose.

    WebXR defines the wrist joint orientation directly. Using it avoids
    amplifying independent position noise from several short palm bones. The
    input must be one matrix returned by :func:`webxr_to_ros_matrices`.
    """

    matrix = np.asarray(ros_wrist_matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected WebXR wrist matrix shape (4, 4), received {matrix.shape}.")
    rotation = matrix[:3, :3]
    if not np.all(np.isfinite(rotation)):
        raise ValueError("WebXR wrist rotation contains non-finite values.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("WebXR wrist rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("WebXR wrist rotation is not right-handed.")
    return rotation @ ROS_WEBXR_WRIST_TO_CANONICAL_PALM


def joint_distance(matrices: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(matrices[first, :3, 3] - matrices[second, :3, 3]))


def update_hysteresis(previous: bool, distance: float, on_threshold: float, off_threshold: float) -> bool:
    """Apply distance hysteresis for a pinch-like gesture."""

    if off_threshold < on_threshold:
        raise ValueError("The off threshold must be greater than or equal to the on threshold.")
    threshold = off_threshold if previous else on_threshold
    return distance < threshold


def gripper_aperture(
    matrices: np.ndarray,
    minimum_ratio: float,
    maximum_ratio: float,
) -> tuple[float, float]:
    """Normalize thumb-index separation by palm width.

    Returns ``(normalized_aperture, raw_ratio)``. The ratio makes the gesture
    independent of the operator's hand size and headset distance.
    """

    if maximum_ratio <= minimum_ratio:
        raise ValueError("maximum_ratio must be greater than minimum_ratio.")
    palm_width = joint_distance(matrices, INDEX_METACARPAL, PINKY_METACARPAL)
    thumb_index = joint_distance(matrices, THUMB_TIP, INDEX_TIP)
    if palm_width < 1e-6:
        return 0.0, 0.0
    ratio = thumb_index / palm_width
    normalized = (ratio - minimum_ratio) / (maximum_ratio - minimum_ratio)
    return float(np.clip(normalized, 0.0, 1.0)), float(ratio)
