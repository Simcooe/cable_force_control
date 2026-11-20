"""Build, inspect and drive the independently installed four-cable mechanism."""

import argparse
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from build_plugin import LIB

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assembly import COLORS, DIRECTORY, FIXED_JOINTS, generate
from carriage import SUPPORT_JOINTS
from drive import AXES, PositionDrive, run_demo
from kinematics import plan_pose
from slx import JOINTS, SOLIDS, read_source

MODEL = DIRECTORY / "assembly.xml"


def rotation(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def initialize(model):
    data = mujoco.MjData(model)
    for _ in range(32):
        mujoco.mj_forward(model, data)
    return data


def check_routes(model):
    """Compare ordered arc-length samples, including feed and anchor, to the old native plugin."""
    scene = mujoco.MjvScene(model, maxgeom=6000)
    worst_shape = worst_length = worst_end = 0.0
    with np.load(DIRECTORY / "routing_reference.npz") as reference:
        for pose, (bend, slide) in enumerate(reference["poses"]):
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            data.qpos[model.joint("j16740").qposadr[0]] = bend
            data.qpos[model.joint("j16626").qposadr[0]] = slide
            for _ in range(32):
                mujoco.mj_forward(model, data)
            mujoco.mjv_updateScene(
                model,
                data,
                mujoco.MjvOption(),
                None,
                mujoco.MjvCamera(),
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            for color in COLORS:
                key = f"{color}_{pose}"
                tendon = model.tendon(f"four_seed_{color}").id
                start = model.tendon_adr[tendon]
                stop = start + model.tendon_num[tendon] - 1
                ends = data.site_xpos[model.wrap_objid[[start, stop]]]
                end_error = np.linalg.norm(
                    ends - reference[key + "_ends"], axis=1
                ).max()
                worst_end = max(worst_end, end_error)
                if not np.isfinite(end_error) or end_error > 1e-7:
                    raise ValueError(f"{key}: feed/anchor mismatch {end_error:g} mm")
                lines = [
                    g
                    for g in scene.geoms[: scene.ngeom]
                    if g.type == mujoco.mjtGeom.mjGEOM_LINE
                    and g.objtype == mujoco.mjtObj.mjOBJ_UNKNOWN
                    and g.objid == tendon
                ]
                if not lines:
                    raise ValueError(f"{color}: missing solved route")
                last = lines[-1]
                points = np.array(
                    [g.pos for g in lines] + [last.pos + last.mat[:, 2] * last.size[2]],
                    dtype=float,
                )
                arc = np.r_[
                    0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
                ]
                old = reference[key]
                samples = np.linspace(0, arc[-1], len(old))
                path = np.column_stack(
                    [np.interp(samples, arc, points[:, j]) for j in range(3)]
                )
                error = np.linalg.norm(path - old, axis=1).max()
                state = data.sensor(color + "_state").data
                length_error = abs(state[0] - float(reference[key + "_length"]))
                if (
                    not np.isfinite(state).all()
                    or state[8] != 0
                    or error > 1e-3
                    or length_error > 1e-5
                ):
                    raise ValueError(
                        f"{key}: shape {error:g} mm, length {length_error:g} mm, status {state[8]}"
                    )
                worst_shape = max(worst_shape, error)
                worst_length = max(worst_length, length_error)
    print(
        f"Old baseline, four cables at three poses: max sampled path error {worst_shape:.3g} mm, length error {worst_length:.3g} mm, feed/anchor error {worst_end:.3g} mm"
    )


def check(model, data, compare_reference):
    source = json.loads((DIRECTORY / "simscape.json").read_text())
    blocks, _ = read_source()
    if (
        model.nplugin != len(COLORS)
        or model.nu != 3
        or model.njnt
        != len(SUPPORT_JOINTS)
        + sum(
            k in JOINTS and sid not in FIXED_JOINTS for sid, (k, _, _) in blocks.items()
        )
    ):
        raise ValueError("Missing source joints or cable instances")
    pos_errors, rot_errors = [], []
    for sid, (kind, name, _) in blocks.items():
        if kind in SOLIDS:
            gid = model.geom(f"{sid}_{name}").id
            mesh = model.geom_dataid[gid]
            rot = data.geom_xmat[gid].reshape(3, 3) @ rotation(model.mesh_quat[mesh]).T
            pos = data.geom_xpos[gid] - rot @ model.mesh_pos[mesh]
        elif kind in ("Pulley", "Belt-Cable Spool", "Belt-Cable End"):
            site = data.site("frame_" + sid)
            pos, rot = site.xpos, site.xmat.reshape(3, 3)
        elif kind in JOINTS:
            bid = model.body("body_" + sid).id
            pos, rot = data.xpos[bid], data.xmat[bid].reshape(3, 3)
            # Reconstruct the parent-side frame from the actual joint coordinate.
            base = np.array(source["poses_m"]["s" + sid + "_base"])
            q = source["signals"]["s" + sid + "_position"]
            base_pos, base_rot = pos, rot
            if kind == "Prismatic Joint":
                base_pos = pos - 1000 * q * rot[:, 2]
            else:
                c, sn = np.cos(q), np.sin(q)
                base_rot = rot @ np.array([[c, sn, 0], [-sn, c, 0], [0, 0, 1]])
            pos_errors.append(np.linalg.norm(base_pos - 1000 * base[:3, 3]))
            rot_errors.append(np.linalg.norm(base_rot - base[:3, :3]))
        else:
            continue
        expected = np.array(source["poses_m"]["s" + sid])
        pos_errors.append(np.linalg.norm(pos - 1000 * expected[:3, 3]))
        rot_errors.append(np.linalg.norm(rot - expected[:3, :3]))
    print(
        f"{model.njnt} joints, {len(pos_errors)} measured frames; "
        f"max pose error {max(pos_errors):.3g} mm / {max(rot_errors):.3g} (rotation matrix norm)"
    )
    if max(pos_errors) > 1e-4 or max(rot_errors) > 1e-6:
        raise ValueError("Assembly poses differ from Simscape")
    if compare_reference:
        check_routes(model)
        # Use the independently recorded single-joint poses to check axis choice and sign.
        planner_drive = PositionDrive(model, data)
        with np.load(DIRECTORY / "routing_reference.npz") as reference:
            for i, (bend, slide) in enumerate(reference["poses"][1:], start=1):
                pose = plan_pose(planner_drive, float(np.rad2deg(bend)), float(-slide))
                changes = np.array(
                    [
                        reference[f"{c}_0_length"] - reference[f"{c}_{i}_length"]
                        for c in ("blue", "green", "purple")
                    ]
                )
                actual = pose.motor_goal * planner_drive.mm_per_rad[[0, 1, 3]]
                if not np.allclose(actual, changes, atol=1e-6, rtol=0):
                    raise ValueError(
                        "Geometric motor goals differ from the reference installation"
                    )
        print("Bend-only and extension-only geometry plans passed.")
    hold = PositionDrive(model, data).move(np.zeros(3), 0.006, hold=0)
    if compare_reference:
        load = hold["carriage_contact"]
        if (
            not load["active_steps"]
            or not abs(load["signed_peak_axial_friction_N"]) > 0
        ):
            raise ValueError(
                "Slide/sleeve check did not develop an axial contact friction force"
            )
        print(
            f"6 ms hold: {load['active_steps']} steps in contact, "
            f"peak axial friction {load['signed_peak_axial_friction_N']:.6f} N"
        )
    print("Three-motor zero-position hold passed.")


def main(action, targets=None, duration=None, pose=None, plan_only=False, run=None):
    if not LIB.exists():
        from build_plugin import build

        build()
    mujoco.mj_loadPluginLibrary(str(LIB))
    if action in ("build", "check"):
        xml = generate()
        # Validate the in-memory candidate before replacing a usable model.
        spec = mujoco.MjSpec.from_string(
            xml,
            include={
                "carriage_hulls.xml": (DIRECTORY / "carriage_hulls.xml").read_bytes()
            },
        )
        spec.meshdir = str(DIRECTORY / "meshes")
        model = spec.compile()
        data = initialize(model)
        check(model, data, compare_reference=(action == "check"))
        if action == "build":
            with TemporaryDirectory(dir=DIRECTORY) as stage:
                temp = Path(stage) / MODEL.name
                temp.write_text(xml + "\n")
                temp.replace(MODEL)
        elif MODEL.read_text() != xml + "\n":
            raise ValueError("Saved MJCF differs. Run build.")
    elif action in ("pose", "move", "demo"):
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        drive = PositionDrive(model, initialize(model))
        if action == "pose":
            try:
                plan = plan_pose(drive, *pose, duration=duration)
            except ValueError as exc:
                raise SystemExit(f"关节目标规划失败：{exc}") from exc
            report = {"plan": plan.diagnostics}
            if not plan_only:
                print(
                    f"几何规划完成，运动时长 {plan.diagnostics['duration_s']:.3f} s，"
                    f"三轴目标角（度）{plan.diagnostics['motor_goal_deg']}",
                    file=sys.stderr,
                    flush=True,
                )
                from playback import Recording, model_hash

                recording = Recording(model.opt.timestep)
                start = time.monotonic()
                motion = drive.move(plan.motor_goal, duration, sample=recording.sample)
                elapsed = time.monotonic() - start
                recording.sample(drive, final=True)
                actual_bend = motion["final_bend_deg"]
                actual_extension = -motion["final_slide_mm"]
                report.update(
                    status="completed",
                    wall_s=elapsed,
                    input_sha256={"model/assembly.xml": model_hash(MODEL)},
                    motion=motion,
                    actual_bend_deg=actual_bend,
                    actual_extension_mm=actual_extension,
                    bend_error_deg=actual_bend - pose[0],
                    extension_error_mm=actual_extension - pose[1],
                )
                run_dir = recording.save(MODEL, report)
                print(f"MuJoCo 回放目录：{run_dir}", file=sys.stderr)
            print(json.dumps(report, indent=2))
        elif action == "move":
            print(json.dumps(drive.move(np.deg2rad(targets), duration), indent=2))
        else:
            print(json.dumps(run_demo(drive), indent=2))
    elif action == "replay":
        from replay import open_replay

        open_replay(MODEL, LIB, initialize)
    elif action == "playback":
        from playback import play

        play(MODEL, run)
    elif action == "view":
        from mujoco import viewer as mjviewer

        model = mujoco.MjModel.from_xml_path(str(MODEL))
        data = initialize(model)
        preset = [1]

        def key_pressed(key):
            if key in (ord("1"), ord("2")):
                preset[:] = [key - ord("0")]

        with mjviewer.launch_passive(model, data, key_callback=key_pressed) as viewer:
            while viewer.is_running():
                if preset:
                    overview = preset.pop() == 1
                    with viewer.lock():
                        viewer.cam.lookat[:] = (
                            model.stat.center
                            if overview
                            else data.site("frame_16615").xpos
                        )
                        viewer.cam.distance = (
                            1.55 * model.stat.extent if overview else 85
                        )
                        viewer.cam.azimuth = 120 if overview else 82
                        viewer.cam.elevation = -16 if overview else -8
                viewer.sync()
                time.sleep(0.033)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("build", "check", "view", "demo", "replay"):
        commands.add_parser(name)
    playback = commands.add_parser(
        "playback", help="MuJoCo native viewer for a saved pose recording"
    )
    playback.add_argument("--run", type=Path, help="结果目录；默认打开最新完整记录")
    move = commands.add_parser(
        "move", help="Command absolute shaft angles from the assembly home pose"
    )
    for axis in AXES:
        move.add_argument("--" + axis, type=float, metavar="DEG", required=True)
    move.add_argument("--duration", type=float, metavar="SECONDS", required=True)
    pose_cmd = commands.add_parser(
        "pose",
        help="Plan and execute absolute mechanism bend and extension from assembly home",
    )
    pose_cmd.add_argument(
        "--angle", type=float, required=True, metavar="DEG", help="机构目标弯折角（度）"
    )
    pose_cmd.add_argument(
        "--length",
        type=float,
        required=True,
        metavar="MM",
        help="相对零位的目标伸出量（mm），伸出为正",
    )
    pose_cmd.add_argument(
        "--duration",
        type=float,
        metavar="SECONDS",
        help="运动时长，默认按三轴共同限速限加速度计算",
    )
    pose_cmd.add_argument(
        "--plan-only", action="store_true", help="只求几何轨迹和三轴终点"
    )
    args = parser.parse_args()
    main(
        args.action,
        [getattr(args, a, None) for a in AXES],
        getattr(args, "duration", None),
        (getattr(args, "angle", None), getattr(args, "length", None)),
        getattr(args, "plan_only", False),
        getattr(args, "run", None),
    )
