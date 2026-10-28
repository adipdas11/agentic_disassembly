from __future__ import annotations

import math
from typing import Iterable


# Bottle-grasp alignment for a parallel-jaw gripper. The palm faces one jaw,
# so thumb-index opposition follows the gripper's jaw-separation axis:
# hand +X (wrist to fingers) -> TCP -Z (gripper approach)
# hand +Y (across the palm)  -> TCP -X
# hand +Z (palm normal)      -> TCP +Y (jaw separation)
DEFAULT_HAND_TO_TOOL_ROTATION = [
    0.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    -1.0,
    0.0,
    0.0,
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def vec_add(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [float(x) + float(y) for x, y in zip(a, b)]


def vec_sub(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def vec_scale(a: Iterable[float], scalar: float) -> list[float]:
    return [float(x) * float(scalar) for x in a]


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def cross(a: Iterable[float], b: Iterable[float]) -> list[float]:
    ax, ay, az = [float(v) for v in a]
    bx, by, bz = [float(v) for v in b]
    return [
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ]


def norm(a: Iterable[float]) -> float:
    return math.sqrt(dot(a, a))


def apply_translation_deadzone(
    delta: Iterable[float],
    deadzone_xyz: Iterable[float],
) -> list[float]:
    """Apply a continuous per-axis dead zone to a 3D translation.

    Values inside the dead zone become zero. Values outside it have the dead
    zone subtracted, avoiding a target-position jump at the threshold.
    """

    values = [float(value) for value in delta]
    thresholds = [float(value) for value in deadzone_xyz]
    if len(values) != 3 or len(thresholds) != 3:
        raise ValueError("translation delta and deadzone must contain exactly three values.")
    if not all(math.isfinite(value) for value in values + thresholds):
        raise ValueError("translation delta and deadzone must contain only finite values.")
    if any(threshold < 0.0 for threshold in thresholds):
        raise ValueError("translation deadzone values must be non-negative.")

    shaped = []
    for value, threshold in zip(values, thresholds):
        magnitude = max(0.0, abs(value) - threshold)
        shaped.append(math.copysign(magnitude, value) if magnitude > 0.0 else 0.0)
    return shaped


def normalize(a: Iterable[float]) -> list[float] | None:
    values = [float(v) for v in a]
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        return None
    magnitude = norm(values)
    if not math.isfinite(magnitude) or magnitude < 1e-8:
        return None
    return [v / magnitude for v in values]


def mat_vec_mul(matrix: list[list[float]], vector: Iterable[float]) -> list[float]:
    vx, vy, vz = [float(v) for v in vector]
    return [
        matrix[0][0] * vx + matrix[0][1] * vy + matrix[0][2] * vz,
        matrix[1][0] * vx + matrix[1][1] * vy + matrix[1][2] * vz,
        matrix[2][0] * vx + matrix[2][1] * vy + matrix[2][2] * vz,
    ]


def quat_normalize(q: Iterable[float]) -> list[float]:
    qx, qy, qz, qw = [float(v) for v in q]
    mag = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if mag < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    return [qx / mag, qy / mag, qz / mag, qw / mag]


def quat_conjugate(q: Iterable[float]) -> list[float]:
    qx, qy, qz, qw = [float(v) for v in q]
    return [-qx, -qy, -qz, qw]


def quat_multiply(a: Iterable[float], b: Iterable[float]) -> list[float]:
    ax, ay, az, aw = [float(v) for v in a]
    bx, by, bz, bw = [float(v) for v in b]
    return quat_normalize(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_rotate_vector(q: Iterable[float], vector: Iterable[float]) -> list[float]:
    """Rotate a vector without changing its magnitude.

    ``quat_multiply`` normalizes its result because it is intended for rotation
    composition. Using it with a pure-vector quaternion would therefore turn
    every non-zero vector into a unit vector.
    """

    values = [float(value) for value in vector]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("rotation vector input must contain three finite values.")
    return mat_vec_mul(quat_to_matrix(q), values)


def pose_pivot_position(
    pose_position: Iterable[float],
    pose_quaternion: Iterable[float],
    pose_to_pivot_xyz: Iterable[float],
) -> list[float]:
    """Return a rigid pose's pivot position in the reference frame."""

    return vec_add(pose_position, quat_rotate_vector(pose_quaternion, pose_to_pivot_xyz))


def pose_position_from_pivot(
    pivot_position: Iterable[float],
    pose_quaternion: Iterable[float],
    pose_to_pivot_xyz: Iterable[float],
) -> list[float]:
    """Recover a pose origin while holding its reference-frame pivot fixed."""

    return vec_sub(pivot_position, quat_rotate_vector(pose_quaternion, pose_to_pivot_xyz))


def quat_to_matrix(q: Iterable[float]) -> list[list[float]]:
    x, y, z, w = quat_normalize(q)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def matrix_to_quat(matrix: list[list[float]]) -> list[float]:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return quat_normalize([qx, qy, qz, qw])


def palm_frame_matrix(
    wrist: Iterable[float],
    index_mcp: Iterable[float],
    middle_mcp: Iterable[float],
    pinky_mcp: Iterable[float],
    handedness: str,
) -> list[list[float]] | None:
    """Build a right-handed anatomical palm frame from four landmarks.

    The returned rotation matrix has the frame axes as columns. Its local X
    points from the wrist to the middle MCP, local Y runs across the palm, and
    local Z is the outward palm normal. Mirroring the lateral landmark order
    for the left hand prevents left/right frame flips while preserving a
    right-handed frame.

    ``None`` is returned for non-finite, coincident, or collinear landmarks.
    """

    hand = str(handedness).strip().lower()
    if hand not in ("left", "right"):
        raise ValueError("handedness must be 'left' or 'right'.")

    points: list[list[float]] = []
    for point in (wrist, index_mcp, middle_mcp, pinky_mcp):
        try:
            values = [float(v) for v in point]
        except (TypeError, ValueError):
            return None
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            return None
        points.append(values)
    wrist_point, index_point, middle_point, pinky_point = points

    longitudinal_hint = vec_sub(middle_point, wrist_point)
    if norm(longitudinal_hint) < 1e-6:
        return None
    x_axis = normalize(longitudinal_hint)
    if x_axis is None:
        return None

    # For a right hand, pinky -> index selects the lateral sign that makes
    # X x Y point out through the palm rather than out through the back of the
    # hand. Mirror that ordering for a left hand so equivalent left/right palm
    # poses produce the same spatial frame while remaining right-handed.
    lateral_hint = vec_sub(index_point, pinky_point)
    if hand == "left":
        lateral_hint = vec_scale(lateral_hint, -1.0)
    lateral_orthogonal = vec_sub(lateral_hint, vec_scale(x_axis, dot(lateral_hint, x_axis)))
    lateral_magnitude = norm(lateral_hint)
    if lateral_magnitude < 1e-6 or norm(lateral_orthogonal) / lateral_magnitude < 0.05:
        return None
    y_axis = normalize(lateral_orthogonal)
    if y_axis is None:
        return None

    z_axis = normalize(cross(x_axis, y_axis))
    if z_axis is None:
        return None
    # Recompute Y to remove the last of the landmark noise while retaining the
    # handedness-selected sign. This also guarantees X x Y = Z.
    y_axis = normalize(cross(z_axis, x_axis))
    if y_axis is None:
        return None

    return [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]],
    ]


def palm_frame_quaternion(
    wrist: Iterable[float],
    index_mcp: Iterable[float],
    middle_mcp: Iterable[float],
    pinky_mcp: Iterable[float],
    handedness: str,
) -> list[float] | None:
    """Return the canonical palm-frame quaternion, or ``None`` if degenerate."""

    matrix = palm_frame_matrix(wrist, index_mcp, middle_mcp, pinky_mcp, handedness)
    return None if matrix is None else matrix_to_quat(matrix)


def validate_rotation_matrix(
    values: Iterable[float] | Iterable[Iterable[float]],
    parameter_name: str = "rotation",
    tolerance: float = 1e-5,
) -> list[list[float]]:
    """Validate and return a finite, right-handed 3x3 rotation matrix."""

    raw = list(values)
    if len(raw) == 3 and all(hasattr(row, "__iter__") for row in raw):
        rows = [[float(v) for v in row] for row in raw]
    else:
        flat = [float(v) for v in raw]
        if len(flat) != 9:
            raise ValueError(f"{parameter_name} must contain exactly 9 row-major values.")
        rows = [flat[0:3], flat[3:6], flat[6:9]]

    if any(len(row) != 3 for row in rows):
        raise ValueError(f"{parameter_name} must be a 3x3 matrix.")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{parameter_name} must contain only finite values.")
    for index, row in enumerate(rows):
        if abs(norm(row) - 1.0) > tolerance:
            raise ValueError(f"{parameter_name} row {index} is not unit length.")
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if abs(dot(rows[first], rows[second])) > tolerance:
            raise ValueError(f"{parameter_name} rows are not orthogonal.")
    determinant = dot(rows[0], cross(rows[1], rows[2]))
    if abs(determinant - 1.0) > tolerance:
        raise ValueError(f"{parameter_name} must be right-handed with determinant +1.")
    return rows


def quat_to_rpy(q: Iterable[float]) -> tuple[float, float, float]:
    """Convert quaternion [qx, qy, qz, qw] to (roll, pitch, yaw) in radians (ZYX convention)."""
    qx, qy, qz, qw = quat_normalize(q)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convert (roll, pitch, yaw) in radians (ZYX convention) to quaternion [qx, qy, qz, qw]."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return quat_normalize([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def quat_to_rotation_vector(q: Iterable[float]) -> list[float]:
    """Convert a quaternion to its shortest axis-angle rotation vector."""

    qx, qy, qz, qw = quat_normalize(q)
    # q and -q represent the same rotation. Select the hemisphere whose angle
    # is at most pi so filtering stays continuous through ordinary wrist motion.
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    vector_magnitude = math.sqrt(qx * qx + qy * qy + qz * qz)
    if vector_magnitude < 1e-8:
        return [2.0 * qx, 2.0 * qy, 2.0 * qz]
    angle = 2.0 * math.atan2(vector_magnitude, clamp(qw, -1.0, 1.0))
    scale = angle / vector_magnitude
    return [qx * scale, qy * scale, qz * scale]


def rotation_vector_to_quat(rotation_vector: Iterable[float]) -> list[float]:
    """Convert an axis-angle rotation vector to [qx, qy, qz, qw]."""

    rx, ry, rz = [float(value) for value in rotation_vector]
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-8:
        return quat_normalize([0.5 * rx, 0.5 * ry, 0.5 * rz, 1.0])
    scale = math.sin(angle * 0.5) / angle
    return quat_normalize([rx * scale, ry * scale, rz * scale, math.cos(angle * 0.5)])


def retarget_hand_orientation(
    origin_hand_quat: Iterable[float],
    current_hand_quat: Iterable[float],
    origin_tool_quat: Iterable[float],
    hand_to_tool_rotation: Iterable[float] | Iterable[Iterable[float]],
    *,
    track_orientation: bool = False,
    track_roll: bool = False,
    track_pitch: bool = False,
    track_yaw: bool = False,
    orientation_scale_xyz: Iterable[float] = (1.0, 1.0, 1.0),
) -> list[float]:
    """Map calibrated hand-local rotation into the calibrated TCP frame.

    The hand delta is ``H0^-1 * H``. Selected anatomical rotation-vector
    components are scaled, conjugated through the hand-to-tool alignment ``A``,
    and applied locally as ``T0 * (A * delta * A^-1)``. Rotation-vector
    filtering avoids the false roll and gimbal coupling produced by Euler
    decomposition during large combined pitch/yaw wrist motions.
    """

    origin_tool = quat_normalize(origin_tool_quat)
    if not (track_orientation or track_roll or track_pitch or track_yaw):
        return origin_tool

    origin_hand = quat_normalize(origin_hand_quat)
    current_hand = quat_normalize(current_hand_quat)
    hand_delta = quat_multiply(quat_conjugate(origin_hand), current_hand)
    scales = [float(value) for value in orientation_scale_xyz]
    if len(scales) != 3 or not all(math.isfinite(value) for value in scales):
        raise ValueError("orientation_scale_xyz must contain three finite values.")
    if any(value < 0.0 for value in scales):
        raise ValueError("orientation_scale_xyz values must be non-negative.")

    roll, pitch, yaw = quat_to_rotation_vector(hand_delta)
    hand_delta = rotation_vector_to_quat([
        roll * scales[0] if track_orientation or track_roll else 0.0,
        pitch * scales[1] if track_orientation or track_pitch else 0.0,
        yaw * scales[2] if track_orientation or track_yaw else 0.0,
    ])

    alignment = matrix_to_quat(validate_rotation_matrix(hand_to_tool_rotation, "hand_to_tool_rotation"))
    tool_delta = quat_multiply(quat_multiply(alignment, hand_delta), quat_conjugate(alignment))
    return quat_multiply(origin_tool, tool_delta)


def quaternion_slerp(q0: Iterable[float], q1: Iterable[float], alpha: float) -> list[float]:
    ax, ay, az, aw = quat_normalize(q0)
    bx, by, bz, bw = quat_normalize(q1)
    cos_half_theta = ax * bx + ay * by + az * bz + aw * bw
    if cos_half_theta < 0.0:
        bx, by, bz, bw = -bx, -by, -bz, -bw
        cos_half_theta = -cos_half_theta
    if cos_half_theta > 0.9995:
        return quat_normalize(
            [
                ax + alpha * (bx - ax),
                ay + alpha * (by - ay),
                az + alpha * (bz - az),
                aw + alpha * (bw - aw),
            ]
        )
    half_theta = math.acos(clamp(cos_half_theta, -1.0, 1.0))
    sin_half_theta = math.sqrt(max(1.0 - cos_half_theta * cos_half_theta, 0.0))
    if sin_half_theta < 1e-6:
        return [ax, ay, az, aw]
    ratio_a = math.sin((1.0 - alpha) * half_theta) / sin_half_theta
    ratio_b = math.sin(alpha * half_theta) / sin_half_theta
    return quat_normalize(
        [
            ax * ratio_a + bx * ratio_b,
            ay * ratio_a + by * ratio_b,
            az * ratio_a + bz * ratio_b,
            aw * ratio_a + bw * ratio_b,
        ]
    )
