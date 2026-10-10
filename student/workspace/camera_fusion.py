"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

Matrix = np.matrix | np.ndarray

# vi: from fusion_lab.workspace_support import get_tracking_params
from fusion_lab.workspace_support import get_tracking_params


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    transform = np.asarray(sensor.veh_to_sens)
    R = transform[:3, :3]
    t = transform[:3, 3]
    p = np.asarray(x).ravel()[:3]
    p_s = R @ p + t
    
    if not np.all(np.isfinite(p_s)):
        return False
        
    x_s, y_s, z_s = p_s
    
    # Camera specific depth check (checking if f_i exists implies camera)
    if hasattr(sensor, 'f_i'):
        if x_s <= 1e-6:
            return False
            
    theta = np.arctan2(y_s, x_s)
    if isinstance(sensor.fov, (list, tuple, np.ndarray)):
        if not (sensor.fov[0] <= theta <= sensor.fov[1]):
            return False
    else:
        if np.abs(theta) > sensor.fov / 2:
            return False
        
    return True


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    transform = np.asarray(sensor.veh_to_sens)
    R = transform[:3, :3]
    t = transform[:3, 3]
    p = np.asarray(x).ravel()[:3]
    p_s = R @ p + t
    
    if not np.all(np.isfinite(p_s)):
        raise ValueError(f"Coordinates nonfinite: {p_s}")
        
    x_s, y_s, z_s = p_s
    
    if x_s <= 1e-6:
        raise ValueError(f"Depth {x_s} <= 1e-6")
        
    u = sensor.c_i - sensor.f_i * y_s / x_s
    v = sensor.c_j - sensor.f_j * z_s / x_s
    
    return np.matrix([[u], [v]])


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    z_mat = np.matrix([[z[0]], [z[1]]])
    R = np.matrix(np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2]))
    return {"z": z_mat, "R": R, "sensor": sensor}
