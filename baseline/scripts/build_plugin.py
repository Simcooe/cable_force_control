"""Build the macOS plug-in against the active Python environment's MuJoCo."""

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".build" / "libcable_unilateral.dylib"


def build():
    package = Path(mujoco.__file__).resolve().parent
    runtime = package / f"libmujoco.{mujoco.__version__}.dylib"
    sources = ROOT / "third_party" / "mujocable"
    LIB.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=LIB.parent) as stage:
        output = Path(stage) / LIB.name
        subprocess.run(
            [
                "clang++",
                "-std=c++17",
                "-O2",
                "-DNDEBUG",
                "-dynamiclib",
                "-I",
                str(package / "include"),
                *sorted(sources.glob("*.cc")),
                runtime,
                "-Wl,-install_name,@rpath/" + LIB.name,
                "-Wl,-rpath,@loader_path/" + os.path.relpath(package, LIB.parent),
                "-o",
                output,
            ],
            check=True,
        )
        output.replace(LIB)


if __name__ == "__main__":
    build()
