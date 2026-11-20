"""Tessellate only the STEP assets referenced by the selected SLX. Output is in mm."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib"))
import FreeCAD  # noqa: F401, I001 - initializes OpenCascade and resets __main__.
import Part
from pathlib import Path
import hashlib
import struct
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import numpy as np


def convert(source, destination):
    with ZipFile(source) as archive:
        files = {
            Path(p.text).name
            for path in archive.namelist()
            if path.startswith("simulink/systems/") and path.endswith(".xml")
            for p in ET.fromstring(archive.read(path)).findall(
                ".//InstanceData/P[@Name='ExtGeomFileName']"
            )
        }
    destination.mkdir(parents=True, exist_ok=True)
    for filename in sorted(files):
        path = source.parent / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        mesh = destination / (digest + ".msh")
        if mesh.exists():
            continue
        shape = Part.Shape()
        shape.read(str(path))
        vertices, faces = shape.tessellate(0.035)
        if not faces:
            raise ValueError(f"Empty mesh for {filename}")
        # MuJoCo binary mesh. STEP geometry returned by FreeCAD is in mm.
        temp = mesh.with_suffix(".tmp")
        with temp.open("wb") as stream:
            stream.write(struct.pack("<4i", len(vertices), 0, 0, len(faces)))
            np.array([(v.x, v.y, v.z) for v in vertices], dtype="<f4").tofile(stream)
            np.asarray(faces, dtype="<i4").tofile(stream)
        temp.replace(mesh)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    convert(
        base.parent / "structure/model_new_visual_aligned.slx", base / "model/meshes"
    )
