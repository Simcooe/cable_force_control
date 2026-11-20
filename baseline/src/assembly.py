"""Generate the zero-pose assembly in mm, g, s from Simscape measurements."""

import json
import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
import numpy as np
from carriage import add_carriage_contact
from slx import (
    ROOT,
    ROUTING,
    SOURCE,
    digest,
    read_source,
    rigid_groups,
)

DIRECTORY = ROOT / "model"
# Source stage and base are fixed to the test stand. Purple shares the axial hinge.
FIXED_JOINTS = {"11549", "16729", "16739"}
COLORS = ("blue", "green", "orange", "purple")


def numbers(values):
    return " ".join(format(float(v), ".16g") for v in np.ravel(values))


def pose_attrs(pose):
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(pose[:3, :3]).ravel())
    return {"pos": numbers(pose[:3, 3]), "quat": numbers(quaternion)}


def file_inertia(params, values):
    for key, unit in [
        ("Mass", "kg"),
        ("CenterOfMass", "mm"),
        ("MomentsOfInertia", "kg*mm^2"),
        ("ProductsOfInertia", "kg*mm^2"),
    ]:
        if params[key + "Units"] != unit or params["InertiaType"] != "Custom":
            raise ValueError(f"Unsupported File Solid units or inertia for {key}")
    tensor = np.diag(values["MomentsOfInertia"])
    yz, xz, xy = values["ProductsOfInertia"]
    tensor[np.triu_indices(3, 1)] = [xy, xz, yz]
    tensor[np.tril_indices(3, -1)] = [xy, xz, yz]
    return 1000 * values["Mass"], np.array(values["CenterOfMass"]), 1000 * tensor


def annulus(asset, name, params, values):
    profile = np.array(values["RevolutionCrossSection"])
    ri, ro = np.unique(profile[:, 0])
    lo, hi = np.unique(profile[:, 1])
    if (
        params["RevolutionExtent"],
        params["RevolutionCrossSectionUnits"],
        params["DensityUnits"],
    ) != ("Full", "mm", "kg/m^3") or set(map(tuple, profile)) != {
        (r, z) for r in (ri, ro) for z in (lo, hi)
    }:
        raise ValueError("Expected a full rectangular annulus")
    count = 48
    angles = np.arange(count) * 2 * np.pi / count
    verts = [
        [r * np.cos(a), r * np.sin(a), z]
        for z in (lo, hi)
        for r in (ri, ro)
        for a in angles
    ]
    faces = []
    for i in range(count):
        j = (i + 1) % count
        for a, b, c, d in [
            (i, j, count + j, count + i),
            (2 * count + i, 3 * count + i, 3 * count + j, 2 * count + j),
            (count + i, count + j, 3 * count + j, 3 * count + i),
            (i, 2 * count + i, 2 * count + j, j),
        ]:
            faces.extend((a, b, c, a, c, d))
    ET.SubElement(
        asset,
        "mesh",
        name=name,
        vertex=numbers(verts),
        face=" ".join(map(str, faces)),
        inertia="shell",
    )
    h = hi - lo
    mass = values["Density"] * 1e-6 * np.pi * (ro**2 - ri**2) * h
    axial = mass * (ri**2 + ro**2) / 2
    transverse = axial / 2 + mass * h * h / 12
    return (
        mass,
        np.array([0, 0, (hi + lo) / 2]),
        np.diag([transverse, transverse, axial]),
    )


def aggregate(elements):
    mass = sum(m for m, _, _ in elements)
    center = sum(m * x for m, x, _ in elements) / mass
    tensor = np.zeros((3, 3))
    for m, x, inertia in elements:
        arm = x - center
        tensor += inertia + m * (np.dot(arm, arm) * np.eye(3) - np.outer(arm, arm))
    return {
        "mass": numbers(mass),
        "pos": numbers(center),
        "fullinertia": numbers([*tensor.diagonal(), *tensor[np.triu_indices(3, 1)]]),
    }


def generate():
    blocks, nets = read_source()
    snapshot = json.loads((DIRECTORY / "simscape.json").read_text())
    if snapshot["slx_sha256"] != digest(SOURCE) or snapshot[
        "parameters_sha256"
    ] != digest(SOURCE.with_name("model_new_DataFile.m")):
        raise ValueError("Source changed. Re-export the assembly from MATLAB.")
    poses = {key[1:]: np.array(matrix) for key, matrix in snapshot["poses_m"].items()}
    for matrix in poses.values():
        matrix[:3, 3] *= 1000
    source_cad = {item["file"]: item["sha256"] for item in snapshot["cad"]}
    world, groups, children = rigid_groups(blocks, nets)
    model = ET.Element("mujoco", model=SOURCE.stem)
    ET.SubElement(
        model, "compiler", angle="radian", meshdir="meshes", inertiafromgeom="false"
    )
    (gravity,) = [
        p for kind, _, p in blocks.values() if kind == "Mechanism Configuration"
    ]
    assert gravity["GravityVectorUnits"] == "m/s^2"
    g = np.fromstring(gravity["GravityVector"].strip("[]").replace(",", " "), sep=" ")
    option = ET.SubElement(
        model,
        "option",
        gravity=numbers(1000 * g),
        timestep="0.00005",
        integrator="Euler",
        solver="Newton",
        cone="elliptic",
        iterations="100",
        tolerance="1e-11",
    )
    ET.SubElement(
        option,
        "flag",
        eulerdamp="disable",
        filterparent="disable",
        nativeccd="enable",
        multiccd="disable",
    )
    ET.SubElement(model, "size", nuser_site="1")
    default = ET.SubElement(model, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", mass="0")
    ET.SubElement(default, "site", size="0.05", rgba="0 0 0 0")
    asset = ET.SubElement(model, "asset")
    worldbody = ET.SubElement(model, "worldbody")
    visited = set()

    @lru_cache(None)
    def cad(filename):
        key = digest(SOURCE.parent / filename)
        if key != source_cad[filename]:
            raise ValueError(f"{filename} changed since MATLAB measured the CAD frames")
        meshfile = key + ".msh"
        if not (DIRECTORY / "meshes" / meshfile).is_file():
            raise FileNotFoundError(meshfile)
        name = "cad_" + key[:16]
        ET.SubElement(asset, "mesh", name=name, file=meshfile, inertia="shell")
        return name

    def add_group(group, parent, parent_pose, joint=None):
        if group in visited:
            raise ValueError("Rigid graph is not a tree")
        visited.add(group)
        frame = np.eye(4) if joint is None else poses[joint]
        body = ET.SubElement(
            parent,
            "body",
            name="base" if joint is None else "body_" + joint,
            **pose_attrs(np.linalg.solve(parent_pose, frame)),
        )
        if joint is not None and joint not in FIXED_JOINTS:
            kind, name, p = blocks[joint]
            if (
                float(p["SpringStiffness"])
                or float(p["DampingCoefficient"])
                or "on" in (p["LowerLimitSpecify"], p["UpperLimitSpecify"])
            ):
                raise ValueError(f"Unsupported spring or limit on {name}")
            ET.SubElement(
                body,
                "joint",
                name="j" + joint,
                type="hinge" if kind == "Revolute Joint" else "slide",
                axis="0 0 1",
            )
        inertias = []
        for sid in groups[group]:
            kind, name, p = blocks[sid]
            local = np.linalg.solve(frame, poses[sid])
            if kind in ROUTING:
                ET.SubElement(body, "site", name="frame_" + sid, **pose_attrs(local))
                continue
            values = snapshot["parts"]["s" + sid]
            if kind == "File Solid":
                mesh = cad(p["ExtGeomFileName"])
                mass, com, tensor = file_inertia(p, values)
            else:
                mesh = "ring_" + sid
                mass, com, tensor = annulus(asset, mesh, p, values)
            rotation = local[:3, :3]
            inertias.append(
                (mass, rotation @ com + local[:3, 3], rotation @ tensor @ rotation.T)
            )
            ET.SubElement(
                body,
                "geom",
                name=f"{sid}_{name}",
                type="mesh",
                mesh=mesh,
                **pose_attrs(local),
                rgba=numbers(
                    [*values["GraphicDiffuseColor"][:3], values["GraphicOpacity"]]
                ),
            )
        body.insert(0, ET.Element("inertial", aggregate(inertias)))
        for child_joint, child_group in children[group]:
            add_group(child_group, body, frame, child_joint)

    add_group(world, worldbody, np.eye(4))
    if visited != set(groups):
        raise ValueError("Disconnected source assembly")

    # Mount the second drum rigidly on the axial shaft, preserving its CAD pose.
    carrier = worldbody.find(".//body[@name='body_16729']")
    shaft = carrier.find("body[@name='body_16742']")
    drum = carrier.find("body[@name='body_16739']")
    carrier.remove(drum)
    drum.attrib.update(pose_attrs(np.linalg.solve(poses["16742"], poses["16739"])))
    shaft.append(drum)

    layout = ET.parse(DIRECTORY / "routing.xml").getroot()
    # Numerically identical poses can seed different triangles on a symmetric V-groove.
    # Use the reference representation only for equivalent Simscape transforms.
    for frame in layout.findall("reference_frame"):
        body = worldbody.find(f".//body[@name='{frame.get('body')}']")
        q1, q2 = [np.fromstring(x.get("quat"), sep=" ") for x in (body, frame)]
        p1, p2 = [np.fromstring(x.get("pos"), sep=" ") for x in (body, frame)]
        if (
            np.linalg.norm(p1 - p2) < 1e-9
            and min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)) < 1e-12
        ):
            body.set("pos", frame.get("pos"))
            body.set("quat", frame.get("quat"))
    asset.extend(layout.find("asset"))
    for mount in layout.findall("mount"):
        name = mount.attrib["body"]
        parent = (
            worldbody if name == "world" else worldbody.find(f".//body[@name='{name}']")
        )
        if parent is None:
            raise ValueError(f"Missing routing mount {name}")
        parent.extend(mount)
    for section in ("extension", "tendon", "sensor"):
        model.append(layout.find(section))
    drives = ET.SubElement(model, "actuator")
    for axis in layout.findall("motor"):
        joint = worldbody.find(f".//joint[@name='{axis.get('joint')}']")
        joint.set("armature", axis.attrib["armature"])
        joint.set("damping", axis.attrib["damping"])
        ET.SubElement(
            drives,
            "motor",
            name=axis.attrib["name"],
            joint=joint.attrib["name"],
            gear="-1000000000",
            ctrllimited="false",
            forcelimited="false",
        )
    add_carriage_contact(model)
    ET.indent(model)
    return ET.tostring(model, encoding="unicode")
