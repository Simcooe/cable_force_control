"""Read rigid connectivity and ordered spool-to-end circuits from the SLX archive."""

import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "structure" / "model_new_visual_aligned.slx"
SOLIDS = {"File Solid", "Revolved Solid"}
JOINTS = {"Prismatic Joint", "Revolute Joint"}
ROUTING = {"Pulley", "Belt-Cable Spool", "Belt-Cable End"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source():
    blocks, nets = {}, []
    with ZipFile(SOURCE) as archive:
        for path in archive.namelist():
            if not (path.startswith("simulink/systems/") and path.endswith(".xml")):
                continue
            system = ET.fromstring(archive.read(path))
            boundaries = defaultdict(list)
            for block in system.findall("Block"):
                sid = block.attrib["SID"]
                kind = " ".join(
                    (
                        block.findtext("P[@Name='SourceType']")
                        or block.get("BlockType")
                    ).split()
                )
                name = " ".join(block.get("Name").split())
                params = {
                    p.get("Name"): p.text for p in block.findall("InstanceData/P")
                }
                if kind == "File Solid":
                    # SLX can store full CAD paths. Resolve against the local structure directory.
                    params["ExtGeomFileName"] = Path(params["ExtGeomFileName"]).name
                blocks[sid] = (kind, name, params)
                if kind == "PMIOPort":
                    side = block.findtext("P[@Name='Side']")
                    order = int(block.findtext("P[@Name='Port']", "1"))
                    boundaries[side].append((order, sid))
            for side, ports in boundaries.items():
                parent = Path(path).stem.removeprefix("system_")
                direction = {"Left": "lconn", "Right": "rconn"}[side]
                for index, (_, sid) in enumerate(sorted(ports), 1):
                    nets.append([f"{parent}#{direction}:{index}", f"{sid}#rconn:1"])
            for wire in system.findall("Line[@LineType='Connection']"):
                nets.append(
                    [p.text for p in wire.iter("P") if p.get("Name") in ("Src", "Dst")]
                )
    return blocks, nets


def rigid_groups(blocks, nets):
    parent = {}

    def root(port):
        parent.setdefault(port, port)
        while port != parent[port]:
            parent[port] = parent[parent[port]]
            port = parent[port]
        return port

    def join(ports):
        for p in ports[1:]:
            parent[root(p)] = root(ports[0])

    by_sid = defaultdict(set)
    for net in nets:
        join(net)
        for port in net:
            by_sid[port.split("#")[0]].add(port)
    for sid, (kind, _, _) in blocks.items():
        if kind in SOLIDS or kind == "Rigid Transform":
            join(sorted(by_sid[sid]))

    (world,) = [s for s, (kind, _, _) in blocks.items() if kind == "World Frame"]
    world_group = root(f"{world}#rconn:1")
    groups = defaultdict(list)
    children = defaultdict(list)
    for sid, (kind, _, _) in blocks.items():
        if kind in SOLIDS:
            groups[root(min(by_sid[sid]))].append(sid)
        elif kind in ROUTING:
            groups[root(f"{sid}#lconn:1")].append(sid)
        elif kind in JOINTS:
            children[root(f"{sid}#lconn:1")].append((sid, root(f"{sid}#rconn:1")))
    return world_group, groups, children


def cable_paths(blocks, nets):
    wires = {port: net for net in nets for port in net}
    paths = []
    for spool, (kind, _, _) in sorted(blocks.items()):
        if kind != "Belt-Cable Spool":
            continue
        path = [(spool, 1)]
        out = f"{spool}#rconn:1"
        (properties,) = [
            p.split("#")[0]
            for p in wires[out]
            if blocks[p.split("#")[0]][0] == "Belt-Cable Properties"
        ]
        while blocks[path[-1][0]][0] != "Belt-Cable End":
            (dest,) = [
                p for p in wires[out] if p != out and p.split("#")[0] != properties
            ]
            sid, port = dest.split("#rconn:")
            if sid in [s for s, _ in path] or blocks[sid][0] not in (
                "Pulley",
                "Belt-Cable End",
            ):
                raise ValueError(f"Invalid cable connection at {dest}")
            path.append((sid, int(port)))
            out = f"{sid}#rconn:{3 - int(port)}"
        paths.append((properties, path))
    return paths
