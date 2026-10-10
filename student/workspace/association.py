"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2

# vi: from fusion_lab.workspace_support import get_tracking_params
# vi: from fusion_lab.workspace_loader import load_workspace_module
from fusion_lab.workspace_support import get_tracking_params
from fusion_lab.workspace_loader import load_workspace_module

kalman = load_workspace_module("kalman")


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement."""
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    dist_sq = float((gamma.T @ np.linalg.inv(S) @ gamma).item())
    return dist_sq


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate."""
    params = get_tracking_params()
    threshold = chi2.ppf(params.gating_threshold, sensor.dim_meas)
    return mhd_sq < threshold


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection."""
    num_tracks = len(track_list)
    num_meas = len(meas_list)
    
    cost_matrix = np.full((num_tracks, num_meas), np.inf)
    
    if num_meas == 0 or num_tracks == 0:
        return np.matrix(cost_matrix)
        
    sensor = meas_list[0].sensor
    
    for i, track in enumerate(track_list):
        if not sensor.in_fov(track.x):
            continue
            
        for j, meas in enumerate(meas_list):
            mhd_sq = mahalanobis_distance(track, meas)
            if chi2_gate(mhd_sq, sensor):
                cost_matrix[i, j] = mhd_sq
                
    return np.matrix(cost_matrix)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem."""
    if association_matrix.size == 0 or np.all(np.isinf(association_matrix)):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)
        
    # Find minimum value index
    min_idx = np.unravel_index(np.argmin(association_matrix), association_matrix.shape)
    i, j = min_idx
    
    if np.isinf(association_matrix[i, j]):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)
        
    track = unassigned_tracks[i]
    meas = unassigned_meas[j]
    
    new_tracks = list(unassigned_tracks)
    new_tracks.pop(i)
    
    new_meas = list(unassigned_meas)
    new_meas.pop(j)
    
    # Delete row i and column j
    new_matrix = np.delete(association_matrix, i, axis=0)
    new_matrix = np.delete(new_matrix, j, axis=1)
    
    return track, meas, np.matrix(new_matrix), new_tracks, new_meas


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management."""
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)
    
    if not unassigned_tracks or not unassigned_meas:
        manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
        return
        
    cost_matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)
    
    while True:
        track, meas, cost_matrix, unassigned_tracks, unassigned_meas = pick_next_pair(
            cost_matrix, unassigned_tracks, unassigned_meas
        )
        
        # Check if np.nan was returned
        if isinstance(track, float) and np.isnan(track):
            break
            
        filter_obj.update(track, meas)
        manager.handle_updated_track(track, sensor)
        
    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
