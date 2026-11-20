"""Three ideal shaft-position inputs for the four-cable assembly with carriage contact."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np
from assembly import COLORS, DIRECTORY
from carriage import contact_load

AXES = ("blue", "green", "axial")


@dataclass(frozen=True)
class Shaft:
    q: int
    v: int
    actuator: int
    speed: float
    acceleration: float
    radius: float


class PositionDrive:
    """Public targets are absolute encoder angles in rad; positive axial reels in purple."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.shafts = []
        layout = ET.parse(DIRECTORY / "routing.xml")
        for axis in AXES:
            spec = layout.find(f"motor[@name='{axis}']")
            joint = model.joint(spec.attrib["joint"])
            radius = float(spec.attrib["radius_mm"])
            self.shafts.append(
                Shaft(
                    int(joint.qposadr[0]),
                    int(joint.dofadr[0]),
                    model.actuator(axis).id,
                    float(spec.attrib["speed_mm_s"]) / radius,
                    float(spec.attrib["accel_mm_s2"]) / radius,
                    radius,
                )
            )
        self.q = np.array([s.q for s in self.shafts])
        self.v = np.array([s.v for s in self.shafts])
        self.motor = np.array([s.actuator for s in self.shafts])
        self.gear = model.actuator_gear[self.motor, 0].copy()
        self.sign = np.sign(self.gear)
        self.cable_axes = [0, 1, 2, 2]
        self.mm_per_rad = np.array([self.shafts[i].radius for i in self.cable_axes]) * [
            1,
            1,
            -1,
            1,
        ]
        reserves = []
        for color, axis, rate in zip(
            COLORS, self.cable_axes, self.mm_per_rad, strict=True
        ):
            instance = layout.find(f"extension/plugin/instance[@name='{color}']")
            config = {node.attrib["key"]: node.attrib["value"] for node in instance}
            joint_id = model.actuator_trnid[self.motor[axis], 0]
            sign = {"positive": 1, "negative": -1}[config["spool_reserve_direction"]]
            if config["spool_joint"] != model.joint(joint_id).name or not np.isclose(
                sign * self.sign[axis] * float(config["spool_radius"]), rate
            ):
                raise ValueError(f"{color}: cable/shaft winding mismatch")
            reserves.append(float(config["spool_reserve_length"]))
        self.reserves = np.array(reserves)
        self.unit_forces = np.eye(model.nv)[self.v].copy()
        self.inverse_columns = np.empty_like(self.unit_forces)

    def encoders(self):
        return self.sign * (self.data.qpos[self.q] - self.model.qpos0[self.q])

    def step(self, target, sample=None):
        """Solve motor efforts for next-step shaft positions, then advance the whole plant."""
        m, d = self.model, self.data
        h = m.opt.timestep
        mujoco.mj_step1(m, d)
        if sample is not None:
            sample(
                self
            )  # Current pose and velocity-stage cable sensors, before integration.
        d.ctrl[self.motor] = 0
        mujoco.mj_fwdActuation(m, d)
        mujoco.mj_fwdAcceleration(m, d)
        # On this fixed-base assembly, the three spool branches are mechanically
        # independent of carriage contact at a given pose/velocity. The cables
        # already apply shaft reaction torques through qfrc_passive.
        if d.nefc:
            mujoco.mj_fwdConstraint(m, d)
        acceleration = d.qacc if d.nefc else d.qacc_smooth
        want = (self.sign * (target - self.encoders()) / h - d.qvel[self.v]) / h
        # Rows of M^-1 for unit torques at each shaft include coupling to all free DOFs.
        mujoco.mj_solveM(m, d, self.inverse_columns, self.unit_forces)
        torque = np.linalg.solve(
            self.inverse_columns[:, self.v].T, want - acceleration[self.v]
        )
        control = torque / self.gear
        if not np.isfinite(control).all():
            raise RuntimeError(f"Nonfinite shaft torque (Nm): {control}")
        d.ctrl[self.motor] = control
        mujoco.mj_step2(m, d)
        if (
            not np.isfinite(d.qacc).all()
            or any(d.warning.number)
            or any(d.sensor(c + "_state").data[8] != 0 for c in COLORS)
        ):
            raise RuntimeError(f"Invalid dynamics or routing at {d.time:g} s")
        return d.actuator_force[self.motor].copy()

    def move_steps(self, target_rad, duration=None):
        """Choose the duration of one position command from the installed shaft limits."""
        goal = np.asarray(target_rad, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Expected three finite shaft targets")
        if duration is not None and (not np.isfinite(duration) or duration <= 0):
            raise ValueError("Expected a positive finite duration")
        if np.any(self.reserves + self.mm_per_rad * goal[self.cable_axes] < 0):
            raise ValueError("Requested payout exceeds the installed spool reserve")
        travel = abs(goal - self.encoders())
        minimum = max(
            max(
                1.875 * x / shaft.speed,
                np.sqrt(10 / np.sqrt(3) * x / shaft.acceleration),
            )
            for shaft, x in zip(self.shafts, travel, strict=True)
        )
        h = self.model.opt.timestep
        n = max(1, int(np.ceil((minimum if duration is None else duration) / h)))
        if n * h < minimum * (1 - 1e-12):
            raise ValueError(
                f"Increase duration to at least {minimum:.6f} s for this position command"
            )
        return n

    def move(self, target_rad, duration=None, hold=0.02, sample=None):
        """Issue one set of absolute shaft targets to the rest-to-rest position drive."""
        if not np.isfinite(hold) or hold < 0:
            raise ValueError("Expected a finite nonnegative hold")
        n = self.move_steps(target_rad, duration)
        start = self.encoders().copy()
        delta = np.asarray(target_rad, dtype=float) - start

        def target_at(index):
            u = min(1.0, index / n)
            return start + delta * (u**3 * (10 + u * (-15 + 6 * u)))

        return self.follow(target_at, n, hold, sample)

    def follow(self, target_at, n, hold=0.02, sample=None):
        """Execute absolute encoder references at the physics clock, then hold the endpoint."""
        h = self.model.opt.timestep
        goal = target_at(n)
        if not np.allclose(target_at(0), self.encoders(), rtol=0, atol=1e-10):
            raise ValueError("Trajectory must start at current shaft positions")
        if np.max(np.abs(self.data.qvel[self.v])) > 1e-8:
            raise ValueError(
                "Rest-to-rest position moves must start with stationary shafts"
            )
        contact_steps = max_contacts = 0
        peak_normal = peak_friction = peak_depth = 0.0
        slide_id = self.model.joint("j16626").id
        peak_load_time = None
        peak_torque = np.zeros(3)
        peak_tension = np.zeros(4)
        max_error = 0.0
        largest_coupling_error = 0.0
        response = np.zeros(2)
        d = self.data
        bend, slide = [self.model.joint(j).qposadr[0] for j in ("j16740", "j16626")]
        start_pose = d.qpos[[bend, slide]].copy()
        for i in range(n + int(np.ceil(hold / h))):
            target = target_at(min(i + 1, n))
            peak_torque = np.maximum(peak_torque, np.abs(self.step(target, sample)))
            count, normal, friction, depth = contact_load(self.model, d, slide_id)
            contact_steps += int(count > 0)
            max_contacts = max(max_contacts, count)
            peak_normal = max(peak_normal, normal)
            peak_depth = max(peak_depth, depth)
            if abs(friction) > abs(peak_friction):
                peak_friction = friction
                peak_load_time = float(d.time - h)
            max_error = max(max_error, float(np.max(np.abs(self.encoders() - target))))
            states = np.array([d.sensor(c + "_state").data.copy() for c in COLORS])
            largest_coupling_error = max(
                largest_coupling_error, abs(states[2, 3] + states[3, 3])
            )
            if not np.isfinite(states).all():
                raise RuntimeError(f"Nonfinite cable sensor at {d.time:g} s")
            peak_tension = np.maximum(peak_tension, states[:, 5] / 1e6)
            response = np.maximum(response, np.abs(d.qpos[[bend, slide]] - start_pose))
        mujoco.mj_forward(self.model, d)
        if sample is not None:
            sample(self)
        states = np.array([d.sensor(c + "_state").data.copy() for c in COLORS])
        enc = self.encoders()
        expected = enc[self.cable_axes] * self.mm_per_rad
        if (
            np.max(np.abs(states[:, 3] - expected)) > 1e-8
            or largest_coupling_error > 1e-8
            or max_error > 1e-9
            or np.any(states[:, 8] != 0)
            or not np.isfinite(states).all()
        ):
            raise RuntimeError(
                "Encoder, spool payout, or final route verification failed"
            )
        return {
            "carriage_contact": {
                "active_steps": contact_steps,
                "max_simultaneous_contacts": max_contacts,
                "peak_normal_sum_N": peak_normal,
                "signed_peak_axial_friction_N": peak_friction,
                "peak_friction_time_s": peak_load_time,
                "max_penetration_mm": peak_depth,
            },
            "time_s": float(d.time),
            "target_deg": np.rad2deg(goal).tolist(),
            "encoder_deg": np.rad2deg(enc).tolist(),
            "max_encoder_error_deg": float(np.rad2deg(max_error)),
            "contraction_mm": dict(zip(COLORS, states[:, 3].tolist())),
            "max_axial_sum_error_mm": float(largest_coupling_error),
            "peak_torque_Nm": dict(zip(AXES, peak_torque.tolist())),
            "peak_tension_N": dict(zip(COLORS, peak_tension.tolist())),
            "peak_bend_change_deg": float(np.rad2deg(response[0])),
            "peak_slide_change_mm": float(response[1]),
            "final_bend_deg": float(np.rad2deg(d.qpos[bend])),
            "final_slide_mm": float(d.qpos[slide]),
            "route_status": dict(zip(COLORS, states[:, 8].astype(int).tolist())),
        }


DEMO_GOAL_DEG = (0.10, -0.08, 0.12)
DEMO_TRAVEL_S = 0.28
DEMO_HOLD_S = 0.02


def run_demo(drive, sample=None):
    """One verified trajectory shared by the numeric demo and the visual replay."""
    out = drive.move(np.deg2rad(DEMO_GOAL_DEG), DEMO_TRAVEL_S, DEMO_HOLD_S, sample)
    back = drive.move(np.zeros(3), DEMO_TRAVEL_S, DEMO_HOLD_S, sample)
    if (
        min(out["peak_bend_change_deg"], out["peak_slide_change_mm"]) < 0.001
        or out["contraction_mm"]["purple"] <= 0
        or out["contraction_mm"]["orange"] >= 0
    ):
        raise RuntimeError(
            "Demo did not produce the expected movement and opposed winding"
        )
    return {"outbound": out, "return": back}
