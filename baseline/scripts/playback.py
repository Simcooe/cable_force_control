"""Record rigid-body states and play a completed pose run in the native MuJoCo viewer."""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, SimpleQueue
from tempfile import TemporaryDirectory

import mujoco
import numpy as np


class Recording:
    def __init__(self, timestep):
        self.stride = max(1, round(0.01 / timestep))
        self.frames = []

    def sample(self, drive, *, final=False):
        d = drive.data
        if (final or round(d.time / drive.model.opt.timestep) % self.stride == 0) and (
            not self.frames or d.time > self.frames[-1][0]
        ):
            self.frames.append((float(d.time), d.qpos.copy(), d.qvel.copy()))

    def save(self, model_path, report):
        times, positions, velocities = zip(*self.frames)
        root = model_path.parents[1] / "results"
        root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=root) as staging:
            temp = Path(staging)
            np.savez_compressed(
                temp / "joint_states.npz", time_s=times, qpos=positions, qvel=velocities
            )
            (temp / "summary.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            dest = root / ("pose_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            temp.rename(dest)
        return dest


def model_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def play(model_path, run=None):
    from mujoco import viewer as mjviewer

    if run is None:
        files = (model_path.parents[1] / "results").glob("*/joint_states.npz")
        latest = max(
            (p for p in files if (p.parent / "summary.json").is_file()),
            key=lambda p: p.stat().st_mtime_ns,
            default=None,
        )
        if latest is None:
            raise SystemExit("尚无可回放的关节记录，请先运行 pose。")
        run = latest.parent
    run = Path(run)
    report = json.loads((run / "summary.json").read_text())
    if report.get("status") != "completed" or report["input_sha256"][
        "model/assembly.xml"
    ] != model_hash(model_path):
        raise SystemExit("记录未完成或所用模型与当前 assembly.xml 不一致。")
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    with np.load(run / "joint_states.npz", allow_pickle=False) as archive:
        times, qpos, qvel = (archive[key] for key in ("time_s", "qpos", "qvel"))
    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or qpos.shape != (len(times), m.nq)
        or qvel.shape != (len(times), m.nv)
        or not all(np.isfinite(a).all() for a in (times, qpos, qvel))
        or not np.isclose(times[-1], report["motion"]["time_s"], rtol=0, atol=1e-8)
    ):
        raise ValueError("关节记录的维数或时间轴异常")
    bend, slide = (m.joint(name).qposadr[0] for name in ("j16740", "j16626"))

    def apply_frame(i):
        d.qpos[:] = qpos[i]
        d.qvel[:] = qvel[i]
        d.time = times[i]
        # Rebuild transforms and cable visualization at this recorded pose. No integration.
        mujoco.mj_forward(m, d)

    def rewind():
        mujoco.mj_resetData(m, d)
        for _ in range(32):
            apply_frame(0)

    rewind()
    hinge = d.xanchor[m.joint("j16740").id].copy()
    m.vis.map.znear = 0.00015
    m.geom("16764_Solid42").rgba[3] = 0.16
    keys = SimpleQueue()
    print(
        f"MuJoCo 回放 {run.name}：{len(times)} 帧，{times[-1]:.4f} 秒。P 暂停，R 重播，1 整机，2 前端。",
        flush=True,
    )
    with mjviewer.launch_passive(
        m, d, key_callback=keys.put, show_left_ui=False, show_right_ui=False
    ) as window:
        i, clock, paused, preset = 0, 0.0, False, 2
        last = time.monotonic()
        while window.is_running():
            now = time.monotonic()
            if not paused:
                clock += now - last
            last = now
            restart = clock >= times[-1] + 1
            while True:
                try:
                    key = keys.get_nowait()
                except Empty:
                    break
                if key == ord("P"):
                    paused = not paused
                elif key == ord("R"):
                    restart = True
                    paused = False
                elif key in (ord("1"), ord("2")):
                    preset = key - ord("0")
            with window.lock():
                if restart:
                    rewind()
                    i, clock = 0, 0.0
                # Visit every recorded pose in order to retain the cable routing branch.
                end = int(np.searchsorted(times, clock, side="right") - 1)
                while i < end:
                    i += 1
                    apply_frame(i)
                if preset:
                    front = preset == 2
                    window.cam.lookat[:] = (
                        hinge + [-14, 0, -24] if front else m.stat.center
                    )
                    window.cam.distance = 180 if front else 1.55 * m.stat.extent
                    window.cam.azimuth = 88 if front else 120
                    window.cam.elevation = -5 if front else -16
                    preset = 0
            window.set_texts(
                (
                    None,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    (
                        f"{'Paused' if paused else 'Replay'}  {times[i]:.2f} / {times[-1]:.2f} s\n"
                        f"Bend {np.rad2deg(qpos[i, bend]):+.2f} deg   Extension {-qpos[i, slide]:+.3f} mm\n"
                        "P pause   R restart   1 overview   2 tip"
                    ),
                    None,
                )
            )
            window.sync()
            time.sleep(0.015)
