#!/bin/sh
set -eu
cd "$(dirname "$0")"
case "${1:-replay}" in
    replay) exec .venv/bin/python -B scripts/model.py replay ;;
    view|playback) exec .venv/bin/mjpython -B scripts/model.py "$@" ;;
    check|demo|move|pose) exec .venv/bin/python -B scripts/model.py "$@" ;;
    build) exec .venv/bin/python -B scripts/model.py build ;;
    import)
        /Applications/MATLAB_R2023a.app/bin/matlab -batch "addpath(fullfile(pwd,'scripts')); export_simscape"
        /Applications/FreeCAD.app/Contents/Resources/bin/python -B scripts/convert_meshes.py
        exec .venv/bin/python -B scripts/model.py build ;;
    *) echo 'Usage: ./baseline.command [replay|playback|view|check|build|import|demo|move|pose]' >&2; exit 2 ;;
esac
