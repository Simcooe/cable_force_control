"""Geometric feedforward from bend/extension to one three-shaft position command."""

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np
from assembly import COLORS, DIRECTORY
from drive import AXES


@dataclass
class PosePlan:
    motor_goal: np.ndarray
    diagnostics: dict


def fixed_guide_ends(model):
    """For a cable with all guides grounded, only endpoint motion can change length."""
    layout = ET.parse(DIRECTORY / "routing.xml")
    for index, color in enumerate(COLORS):
        instance = layout.find(f"extension/plugin/instance[@name='{color}']")
        config = {node.attrib["key"]: node.attrib["value"] for node in instance}
        surface_bodies = [
            model.geom(name).bodyid[0] for name in config["wrap_geoms"].split()
        ]
        tendon = model.tendon(config["route_tendon"]).id
        start = model.tendon_adr[tendon]
        sites = model.wrap_objid[start : start + model.tendon_num[tendon]]
        bodies = [*surface_bodies, *model.site_bodyid[sites[1:-1]]]
        if all(model.body_weldid[b] == 0 for b in bodies):
            yield index, sites[[0, -1]]


def plan_pose(drive, angle_deg, length_mm, duration=None):
    """Absolute bend in degrees and extension in mm (+ toward the tip), relative to assembly zero."""
    if not np.isfinite([angle_deg, length_mm]).all():
        raise ValueError("目标角度和伸出量必须为有限数值")
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("运动时长必须是正的有限数值")

    m = drive.model
    # mj_copyData also copies the native cable route history. Queries only change this copy.
    query = copy.copy(drive.data)
    query.qvel[:] = 0
    dofs = [m.joint(j).qposadr[0] for j in ("j16740", "j16626")]
    initial = query.qpos[dofs].copy()
    goal = np.array([np.deg2rad(angle_deg), -length_mm])
    # Continue the native wrap branch for large moves. These are geometry
    # queries on a copy, with no times or motor references attached.
    bend_change, slide_change = np.abs(goal - initial)
    intervals = max(1, int(np.ceil(max(np.rad2deg(bend_change), slide_change / 0.25))))
    progress = np.linspace(0, 1, intervals + 1)
    sensors = [query.sensor(color + "_state").data for color in COLORS]
    previous_lengths = None
    fixed = dict(fixed_guide_ends(m))
    # With other joints frozen, the endpoint speed per unit progress is bounded
    # by |J_bend|*|delta_bend| + |J_slide|*|delta_slide|. The radius to the
    # bend axis is convex along the linear slide path, so endpoint maxima bound
    # each interval. A whole extra wrap cannot pass this continuity check.
    previous_rates = {}
    vel_dofs = [m.joint(j).dofadr[0] for j in ("j16740", "j16626")]
    jac = np.empty((3, m.nv))
    for i, u in enumerate(progress):
        query.qpos[dofs] = initial + u * (goal - initial)
        for _ in range(32):
            mujoco.mj_forward(m, query)
            if all(sensor[8] == 0 for sensor in sensors):
                break
        else:
            raise ValueError(f"绳路在目标路径 {u:.1%} 处无有效解")
        lengths = np.array([sensor[0] for sensor in sensors])
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("插件返回无效绳长")
        if i == 0:
            start_lengths = lengths
        for k, ends in fixed.items():
            rate = 0.0
            for site in ends:
                mujoco.mj_jacSite(m, query, jac, None, site)
                rate += np.linalg.norm(jac[:, vel_dofs], axis=0) @ np.abs(
                    goal - initial
                )
            if i:
                bound = max(rate, previous_rates[k]) * (u - progress[i - 1])
                change = abs(lengths[k] - previous_lengths[k])
                if change > bound + 0.001:  # Allow 1 µm numerical route discretization.
                    raise ValueError(
                        f"{COLORS[k]} 绳路不连续：弯折 {np.rad2deg(initial[0] + progress[i - 1] * (goal[0] - initial[0])):.3f}° "
                        f"到 {np.rad2deg(query.qpos[dofs[0]]):.3f}°，路径变化 {change:.5f} mm，"
                        f"该区间端点行程上界 {bound:.5f} mm。检查导向接触分支。"
                    )
            previous_rates[k] = rate
        previous_lengths = lengths

    winding = start_lengths - lengths
    chosen = [COLORS.index(c) for c in ("blue", "green", "purple")]
    motor_goal = drive.encoders() + winding[chosen] / drive.mm_per_rad[chosen]
    steps = drive.move_steps(motor_goal, duration)
    orange = COLORS.index("orange")
    axial_travel = motor_goal[2] - drive.encoders()[2]
    info = {
        "target_bend_deg": angle_deg,
        "target_extension_mm": length_mm,
        "duration_s": float(steps * m.opt.timestep),
        "geometric_path_change_mm": dict(zip(COLORS, (-winding).tolist())),
        "motor_goal_deg": dict(zip(AXES, np.rad2deg(motor_goal).tolist())),
        "orange_geometric_vs_spool_difference_mm": float(
            winding[orange] - axial_travel * drive.mm_per_rad[orange]
        ),
    }
    return PosePlan(motor_goal, info)
