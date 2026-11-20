"""Carriage/sleeve contact geometry and the tangential load on the moving group."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

SLEEVE, CARRIAGE = 32, 64
SUPPORT_JOINTS = (
    "carriage_lateral_x",
    "carriage_lateral_y",
    "carriage_tilt_x",
    "carriage_tilt_y",
)


def add_carriage_contact(model):
    defaults = ET.SubElement(
        model.find("default"), "default", {"class": "carriage_surface"}
    )
    ET.SubElement(
        defaults,
        "geom",
        type="mesh",
        mass="0",
        group="4",
        rgba="0 0 0 0",
        condim="3",
        friction="0.2 0 0",
        solref=".0003 1",
        solimp=".98 .99 .001",
        margin="-.015",
        gap=".015",
    )
    ET.SubElement(model.find("asset"), "include", file="carriage_hulls.xml")
    carriage = model.find(".//body[@name='body_16626']")
    carriage.find("joint[@name='j16626']").set("damping", "350")
    # These four passive coordinates permit contact loads from misalignment and tilt.
    for name, kind, axis in zip(
        SUPPORT_JOINTS,
        ("slide", "slide", "hinge", "hinge"),
        ("1 0 0", "0 1 0", "1 0 0", "0 1 0"),
        strict=True,
    ):
        ET.SubElement(
            carriage, "joint", name=name, type=kind, axis=axis, pos="0 -1.25 -10"
        )

    for sid, solid, count, mesh, moving in (
        ("16767", "Solid46", 672, "cad_6842d9d4996e274f", False),
        ("16757", "Solid34", 42, "cad_fbfeb801a93a8578", True),
        ("16764", "Solid42", 367, "cad_b886811d232e1dec", True),
    ):
        owner = carriage if moving else model.find(".//body[@name='body_16740']")
        visual = owner.find(f"geom[@name='{sid}_{solid}']")
        if visual is None or visual.get("mesh") != mesh:
            raise ValueError(
                f"{solid} CAD changed; regenerate its convex contact pieces"
            )
        for i in range(count):
            name = f"carriage_collision_{solid.lower()}_{i:03d}"
            ET.SubElement(
                owner,
                "geom",
                {"class": "carriage_surface"},
                name=name,
                mesh=name,
                pos=visual.attrib["pos"],
                quat=visual.attrib["quat"],
                contype=str(CARRIAGE if moving else SLEEVE),
                conaffinity=str(SLEEVE if moving else CARRIAGE),
            )


def contact_load(model, data, slide_id):
    """Sum tangential force on the carriage projected along positive slide (retraction)."""
    count = 0
    normal = axial = penetration = 0.0
    force = np.empty(6)
    axis = data.xaxis[slide_id]
    for i in np.flatnonzero(data.contact.efc_address >= 0):
        contact = data.contact[i]
        a, b = map(int, contact.geom)
        t1, t2 = model.geom_contype[a], model.geom_contype[b]
        if (t1, t2) not in ((SLEEVE, CARRIAGE), (CARRIAGE, SLEEVE)):
            continue
        mujoco.mj_contactForce(model, data, i, force)
        count += 1
        normal += force[0]
        # Contact frame rows are world-space basis vectors; native wrench acts on geom2.
        tangent = force[1:3] @ contact.frame.reshape(3, 3)[1:]
        axial += (1 if t2 == CARRIAGE else -1) * np.dot(tangent, axis)
        penetration = max(penetration, -contact.dist)
    return count, normal / 1e6, axial / 1e6, penetration
