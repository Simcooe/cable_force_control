"""Render the validated motion to a local movie and play it in the default browser."""

import hashlib
import math
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import mujoco
import numpy as np
from drive import DEMO_HOLD_S, DEMO_TRAVEL_S, PositionDrive, run_demo
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
FPS = 30
SAMPLE_S = 0.0025
MAG = 200
BG, PANEL, TEXT, MUTED = "#101821", "#1a2632", "#f1f5f8", "#a9b9c8"
ORANGE, PURPLE, TEAL = "#ff994f", "#cb91ff", "#65e1cd"
ROOT = Path(__file__).resolve().parents[1]


@lru_cache(None)
def font(size, mono=False):
    filename = "Menlo.ttc" if mono else "STHeiti Light.ttc"
    return ImageFont.truetype("/System/Library/Fonts/" + filename, size)


def label(draw, xy, string, size, fill=TEXT, mono=False, anchor="la"):
    draw.text(xy, string, font=font(size, mono), fill=fill, anchor=anchor)


def signed(value, digits):
    return (
        f"{value:+.{digits}f}" if abs(value) >= 0.5 * 10**-digits else f"{0:.{digits}f}"
    )


def schematic(draw, bend, extension):
    """Idealized side view. Apply magnification to measured deltas only."""
    pivot = np.array([492.0, 481.0])
    theta = MAG * bend
    axis = np.array([-math.cos(theta), math.sin(theta)])
    normal = np.array([math.sin(theta), math.cos(theta)])

    def rod(lo, hi, radius, color):
        coords = [
            tuple(pivot + x * axis + y * normal)
            for x, y in ((lo, -radius), (hi, -radius), (hi, radius), (lo, radius))
        ]
        draw.polygon(coords, fill=color)

    # Grey outline marks the zero-pose moving member, behind the colored schematic.
    draw.rounded_rectangle((226, 473, 395, 489), radius=4, outline=MUTED, width=2)
    draw.line((492, 481, 562, 481), fill=MUTED, width=8)
    travel_px = MAG * extension * 4.0
    rod(65 + travel_px, 266 + travel_px, 6, TEAL)
    rod(0, 150, 12, "#7895a7")
    draw.ellipse((484, 473, 500, 489), fill=BG, outline=TEXT, width=2)
    label(draw, (54, 535), "弯折角", 19, MUTED)
    label(draw, (54, 568), signed(math.degrees(bend), 4) + "°", 29, TEAL, True)
    label(draw, (331, 535), "伸出量", 19, MUTED)
    label(draw, (331, 568), signed(extension * 1000, 2) + " µm", 29, TEAL, True)


def gauges(draw, amounts, velocities):
    zero, span, limit = 971, 174, 25.0  # µm
    for y, name, amount, speed, color in zip(
        (416, 510), ("橙绳", "紫绳"), amounts, velocities, (ORANGE, PURPLE), strict=True
    ):
        direction = "收绳" if speed > 1e-6 else "放绳" if speed < -1e-6 else "保持"
        label(draw, (685, y), name, 22, color)
        label(draw, (768, y), direction, 20, TEXT)
        label(draw, (1211, y), signed(amount, 2) + " µm", 21, color, True, "ra")
        y += 47
        draw.line((zero - span, y, zero + span, y), fill="#405364", width=5)
        draw.line((zero, y - 10, zero, y + 10), fill=MUTED, width=2)
        tip = zero + span * amount / limit
        draw.line((zero, y, tip, y), fill=color, width=9)
        draw.ellipse((tip - 5, y - 5, tip + 5, y + 5), fill=color)
    label(draw, (zero - span, 590), "放绳 −25", 15, MUTED, anchor="ma")
    label(draw, (zero, 590), "0", 15, MUTED, True, "ma")
    label(draw, (zero + span, 590), "收绳 +25", 15, MUTED, anchor="ma")


def render_movie(model_path, destination, initialize):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = initialize(model)
    drive = PositionDrive(model, data)
    # Display-only cutaway of the front end. Collisions and inertias are untouched.
    for i, (x, y, z) in enumerate(data.geom_xpos):
        if abs(x) > 11 or abs(y + 231) > 12 or not 390 < z < 517:
            model.geom_rgba[i, 3] = 0
    model.geom("16764_Solid42").rgba[3] = 0.13
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, -231, 453]
    cam.distance, cam.azimuth, cam.elevation = 128, 90, -6
    model.vis.global_.offwidth, model.vis.global_.offheight = 300, 1208
    model.vis.map.znear = 0.00015
    cycle = DEMO_TRAVEL_S + DEMO_HOLD_S
    total = 2 * cycle
    frame_count = 0
    stride = round(SAMPLE_S / model.opt.timestep)
    bend_idx, slide_idx = (model.joint(j).qposadr[0] for j in ("j16740", "j16626"))
    home_bend, home_slide = data.qpos[[bend_idx, slide_idx]]
    writer = imageio_ffmpeg.write_frames(
        str(destination),
        (WIDTH, HEIGHT),
        fps=FPS,
        codec="libx264",
        quality=8,
        output_params=["-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )
    writer.send(None)
    try:
        with mujoco.Renderer(model, height=1208, width=240, max_geom=6000) as renderer:

            def record(_drive):
                nonlocal frame_count
                step = round(data.time / model.opt.timestep)
                if step != frame_count * stride:
                    return
                t = data.time
                enc = np.rad2deg(drive.encoders())
                amounts = (
                    np.array(
                        [
                            data.sensor(c + "_state").data[3]
                            for c in ("orange", "purple")
                        ]
                    )
                    * 1000
                )
                if not np.allclose(
                    amounts,
                    drive.encoders()[2] * drive.mm_per_rad[2:] * 1000,
                    rtol=0,
                    atol=1e-7,
                ):
                    raise ValueError("Unsynchronized cable readings in replay")
                speed = (
                    drive.sign[2] * data.qvel[drive.v[2]] * drive.mm_per_rad[2:] * 1000
                )
                phase = (
                    "去程"
                    if step < round(DEMO_TRAVEL_S / model.opt.timestep)
                    else "到位保持"
                    if step < round(cycle / model.opt.timestep)
                    else "回程"
                    if step < round((cycle + DEMO_TRAVEL_S) / model.opt.timestep)
                    else "回零保持"
                )
                renderer.update_scene(data, camera=cam)
                front = Image.fromarray(renderer.render()).transpose(
                    Image.Transpose.ROTATE_270
                )
                canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
                canvas.paste(front, (36, 87))
                d = ImageDraw.Draw(canvas)
                label(d, (38, 23), "三轴小动作回放", 34)
                label(d, (340, 32), "前端剖视 · 原比例运动", 20, MUTED)
                label(d, (1000, 30), phase, 22, TEAL, anchor="ra")
                label(d, (1240, 30), f"{t:.3f} / {total:.3f} s", 20, TEXT, True, "ra")
                d.rounded_rectangle((28, 343, 637, 635), 16, fill=PANEL)
                d.rounded_rectangle((653, 343, 1252, 635), 16, fill=PANEL)
                label(d, (54, 365), "弯折与伸缩", 25)
                label(d, (608, 374), f"变化量 ×{MAG} 示意", 16, MUTED, anchor="ra")
                schematic(
                    d,
                    data.qpos[bend_idx] - home_bend,
                    home_slide - data.qpos[slide_idx],
                )
                label(d, (685, 365), "共轴卷筒收放量", 25)
                label(d, (1224, 374), "相对初始位置", 16, MUTED, anchor="ra")
                gauges(d, amounts, speed)
                label(d, (38, 653), "电机角度", 18, MUTED)
                for x, n, val in zip(
                    (194, 481, 769), ("蓝轴", "绿轴", "轴向"), enc, strict=True
                ):
                    label(d, (x, 653), n, 18, MUTED)
                    label(d, (x + 64, 651), signed(val, 3) + "°", 21, TEXT, True)
                label(
                    d, (1240, 655), f"{FPS * SAMPLE_S:g} 倍速", 17, MUTED, anchor="ra"
                )
                d.line((36, 696, 1244, 696), fill="#304558", width=3)
                d.line((36, 696, 36 + 1208 * t / total, 696), fill=TEAL, width=3)
                writer.send(np.asarray(canvas))
                frame_count += 1

            result = run_demo(drive, record)
    finally:
        writer.close()
    if frame_count != round(total / SAMPLE_S) + 1:
        raise RuntimeError(f"Incomplete replay: {frame_count} frames")
    return result


def open_replay(model_path, plugin, initialize):
    directory = ROOT / "replay"
    directory.mkdir(exist_ok=True)
    page = directory / "index.html"
    movie = directory / "demo.mp4"
    fingerprint = hashlib.sha256()
    inputs = [
        model_path,
        plugin,
        ROOT / "model/routing.xml",
        ROOT / "model/carriage_hulls.xml",
        ROOT / "src/drive.py",
        Path(__file__),
        ROOT / "scripts/model.py",
        ROOT / "requirements.txt",
        *sorted((model_path.parent / "meshes").glob("*.msh")),
    ]
    for path in inputs:
        fingerprint.update(hashlib.sha256(path.read_bytes()).digest())
    version = fingerprint.hexdigest()
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>四绳机构 · 三轴小动作回放</title>
<style>html,body{{height:100%;margin:0;background:#101821}}body{{display:grid;place-items:center}}
video{{width:min(100vw,177.778vh);max-height:100vh;display:block}}</style>
<video src="demo.mp4?v={version}" aria-label="三轴小动作回放" autoplay muted playsinline controls loop></video></html>"""
    if (
        not movie.is_file()
        or movie.stat().st_size == 0
        or not page.is_file()
        or page.read_text() != html
    ):
        print("正在计算三轴动作并生成回放…", flush=True)
        with TemporaryDirectory(dir=directory) as stage:
            out = Path(stage) / movie.name
            render_movie(model_path, out, initialize)
            temp_page = Path(stage) / page.name
            temp_page.write_text(html)
            os.replace(out, movie)
            os.replace(temp_page, page)
    subprocess.run(["open", str(page)], check=True)
