#!/usr/bin/env python3
# ============================================================================
# Allen-Cahn-CUDA — Dendrite visualization tool
# ----------------------------------------------------------------------------
# Reads the VTK structured-grid (.vts) snapshots produced by the simulation
# and renders a production-quality PNG for each time step. Designed to run
# on headless servers (Lightning.ai, CI, docker) via PyVista + Xvfb.
#
# Rendering modes
#   iso       phi=0 isosurface (the solid-liquid interface), coloured by
#             dimensionless temperature u. Physically intuitive.
#   slice     three orthogonal slices through the domain, coloured by phi
#             with a diverging RdBu_r cmap. Shows the interior structure.
#   combined  (default) isosurface + orthogonal slices in the same scene.
#             Best picture of the dendrite shape AND the bulk temperature.
#
# Features
#   - Headless rendering (pv.start_xvfb) — no X server required.
#   - Global colormap prescan — the limits of phi and u are computed across
#     ALL snapshots before the first render, so the colour scale is stable
#     (frames can be stitched into a video without flicker).
#   - Natural sort of "output_<step>.vts" — so step 2 comes before step 10.
#   - Step overlay, timestamp, axes gizmo, bounding box.
#   - Optional MP4 stitching via imageio[ffmpeg].
#
# Usage (standalone)
#   python scripts/visualize_dendrite.py \
#       --input-dir ./out --output-dir ./viz \
#       --view combined --make-video --fps 12
#
# Usage (Makefile)
#   make cuda-visualize                       # uses defaults
#   make cuda-visualize VIZ_VIEW=iso
#   make cuda-visualize VIZ_IN=out VIZ_OUT=viz VIZ_FPS=24
#
# Dependencies
#   pyvista >= 0.44, numpy, imageio[ffmpeg] (only for --make-video)
#   xvfb (system package) for headless rendering
# ============================================================================

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: pyvista is required. Install with:\n"
        "  pip install 'pyvista>=0.44' 'imageio[ffmpeg]'\n"
    )
    raise SystemExit(1) from exc


# ── Constants ────────────────────────────────────────────────────────────────

STEP_RE = re.compile(r"output_(\d+)\.vts$")
DEFAULT_ISO_VALUE = 0.0           # phi=0 is the solid-liquid interface
DEFAULT_VIEW = "combined"
DEFAULT_PHI_CMAP = "RdBu_r"        # diverging (phi in [-1, +1])
DEFAULT_U_CMAP = "inferno"         # sequential (u is dimensionless temperature)
DEFAULT_BG = "white"
DEFAULT_WINDOW = (1280, 960)


# ── File discovery ───────────────────────────────────────────────────────────

def natural_step(path: str) -> int:
    m = STEP_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def discover_frames(input_dir: Path, pattern: str) -> List[Path]:
    matches = sorted(glob.glob(str(input_dir / pattern)), key=natural_step)
    return [Path(p) for p in matches if natural_step(p) >= 0]


# ── Global colour-scale prescan ──────────────────────────────────────────────

def compute_global_ranges(frames: List[Path],
                          max_scan: Optional[int] = None
                          ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Walk every snapshot and return ((phi_lo, phi_hi), (u_lo, u_hi)).

    For phi we clamp to [-1, +1] because that is the theoretical range of the
    order parameter; small numerical overshoots are ignored so the colormap is
    symmetric around 0. For u we take the true [min, max] across all frames.
    """
    scanned = 0
    u_lo, u_hi = np.inf, -np.inf
    phi_lo, phi_hi = np.inf, -np.inf

    subset = frames if max_scan is None else frames[:max_scan]
    for f in subset:
        try:
            grid = pv.read(str(f))
        except Exception as exc:
            sys.stderr.write(f"WARN: failed to read {f}: {exc}\n")
            continue
        if "u" in grid.point_data:
            ua = np.asarray(grid.point_data["u"])
            u_lo = min(u_lo, float(ua.min()))
            u_hi = max(u_hi, float(ua.max()))
        if "phi" in grid.point_data:
            pa = np.asarray(grid.point_data["phi"])
            phi_lo = min(phi_lo, float(pa.min()))
            phi_hi = max(phi_hi, float(pa.max()))
        scanned += 1

    if scanned == 0:
        raise RuntimeError("No readable frames found during prescan")

    # phi is physically in [-1, 1] — clamp for symmetry
    phi_lo = max(-1.0, phi_lo if np.isfinite(phi_lo) else -1.0)
    phi_hi = min(+1.0, phi_hi if np.isfinite(phi_hi) else +1.0)

    if not np.isfinite(u_lo) or not np.isfinite(u_hi) or u_lo == u_hi:
        u_lo, u_hi = -1.0, 0.0  # sensible default if file had no 'u'

    # Pad u a touch so the top/bottom colours are not flat
    pad = 0.05 * max(1e-8, u_hi - u_lo)
    return (phi_lo, phi_hi), (u_lo - pad, u_hi + pad)


# ── Rendering primitives ─────────────────────────────────────────────────────

def set_isometric_camera(plotter: pv.Plotter, bounds: Tuple[float, ...]) -> None:
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = bounds
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    cz = 0.5 * (z_lo + z_hi)
    diag = float(np.sqrt((x_hi - x_lo) ** 2
                         + (y_hi - y_lo) ** 2
                         + (z_hi - z_lo) ** 2))
    # Pull the camera 1.8 diagonals away along (1,1,1) for a comfortable iso view
    d = 1.05 * diag
    plotter.camera_position = [
        (cx + d, cy + d, cz + d),  # position
        (cx, cy, cz),               # focal point
        (0, 0, 1),                  # view up
    ]
    plotter.camera.zoom(1.1)


def add_isosurface(plotter: pv.Plotter,
                   grid: pv.StructuredGrid,
                   iso_value: float,
                   u_clim: Tuple[float, float],
                   u_cmap: str) -> None:
    if "phi" not in grid.point_data:
        return
    try:
        iso = grid.contour(isosurfaces=[iso_value], scalars="phi")
    except Exception as exc:
        sys.stderr.write(f"WARN: contour failed: {exc}\n")
        return
    if iso.n_points == 0:
        return  # nothing to draw this frame

    # Colour the isosurface by the local dimensionless temperature u
    if "u" in iso.point_data:
        plotter.add_mesh(
            iso,
            scalars="u",
            cmap=u_cmap,
            clim=u_clim,
            smooth_shading=True,
            ambient=0.25,
            diffuse=0.65,
            specular=0.45,
            specular_power=18,
            name="dendrite",
            scalar_bar_args={
                "title": "u (temperature)",
                "title_font_size": 14,
                "label_font_size": 12,
                "n_labels": 5,
                "position_x": 0.82,
                "position_y": 0.10,
                "width": 0.10,
                "height": 0.55,
            },
        )
    else:
        plotter.add_mesh(iso, color="lightsteelblue", smooth_shading=True,
                         name="dendrite")


def add_orthogonal_slices(plotter: pv.Plotter,
                          grid: pv.StructuredGrid,
                          phi_clim: Tuple[float, float],
                          phi_cmap: str) -> None:
    if "phi" not in grid.point_data:
        return
    try:
        slices = grid.slice_orthogonal()
    except Exception as exc:
        sys.stderr.write(f"WARN: slice_orthogonal failed: {exc}\n")
        return

    plotter.add_mesh(
        slices,
        scalars="phi",
        cmap=phi_cmap,
        clim=phi_clim,
        opacity=0.85,
        name="slices",
        show_scalar_bar=False,
    )


def add_scene_decoration(plotter: pv.Plotter,
                         grid: pv.StructuredGrid,
                         title: str) -> None:
    plotter.add_mesh(grid.outline(), color="black", line_width=1.5)
    plotter.add_axes(interactive=False, line_width=3, labels_off=False)
    plotter.add_text(
        title,
        position="upper_left",
        font_size=12,
        color="black",
        shadow=False,
        name="title",
    )


# ── Main frame rendering ─────────────────────────────────────────────────────

def render_frame(frame_path: Path,
                 png_path: Path,
                 view: str,
                 phi_clim: Tuple[float, float],
                 u_clim: Tuple[float, float],
                 phi_cmap: str,
                 u_cmap: str,
                 iso_value: float,
                 window_size: Tuple[int, int],
                 background: str,
                 total: int,
                 index: int) -> bool:
    """Render a single .vts snapshot to a PNG. Returns True on success."""
    try:
        grid = pv.read(str(frame_path))
    except Exception as exc:
        sys.stderr.write(f"ERROR reading {frame_path}: {exc}\n")
        return False

    step = natural_step(str(frame_path))
    title = (f"Allen-Cahn dendrite\n"
             f"step = {step}   frame {index + 1}/{total}   |   "
             f"grid {grid.dimensions}")

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background(background)

    if view in ("slice", "combined"):
        add_orthogonal_slices(plotter, grid, phi_clim, phi_cmap)
    if view in ("iso", "combined"):
        add_isosurface(plotter, grid, iso_value, u_clim, u_cmap)

    add_scene_decoration(plotter, grid, title)
    set_isometric_camera(plotter, grid.bounds)
    plotter.enable_anti_aliasing("ssaa")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotter.screenshot(str(png_path), return_img=False)
    except Exception as exc:
        sys.stderr.write(f"ERROR writing {png_path}: {exc}\n")
        plotter.close()
        return False
    finally:
        plotter.close()

    return True


# ── Video stitching ──────────────────────────────────────────────────────────

def stitch_video(png_paths: List[Path], mp4_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        sys.stderr.write(
            "WARN: imageio not installed — cannot create video. "
            "Install with: pip install 'imageio[ffmpeg]'\n"
        )
        return

    if not png_paths:
        sys.stderr.write("WARN: no PNGs — skipping video stitching\n")
        return

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.write(f"==> Stitching {len(png_paths)} frames -> {mp4_path} @ {fps} fps\n")
    writer = imageio.get_writer(
        str(mp4_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for p in png_paths:
            writer.append_data(imageio.imread(str(p)))
    finally:
        writer.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="visualize_dendrite",
        description="Render Allen-Cahn .vts snapshots to PNGs (and optionally MP4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, default=Path("out"),
                   help="Directory containing output_*.vts files")
    p.add_argument("--output-dir", type=Path, default=Path("viz"),
                   help="Directory to write frame_*.png (and optional mp4)")
    p.add_argument("--pattern", default="output_*.vts",
                   help="Glob pattern for snapshot files")
    p.add_argument("--view", choices=("iso", "slice", "combined"),
                   default=DEFAULT_VIEW,
                   help="Rendering mode")
    p.add_argument("--iso-value", type=float, default=DEFAULT_ISO_VALUE,
                   help="phi isosurface value (solid-liquid interface)")
    p.add_argument("--phi-cmap", default=DEFAULT_PHI_CMAP,
                   help="Colormap for phi slices")
    p.add_argument("--u-cmap", default=DEFAULT_U_CMAP,
                   help="Colormap for u isosurface")
    p.add_argument("--window-size", nargs=2, type=int, metavar=("W", "H"),
                   default=list(DEFAULT_WINDOW),
                   help="Render window size in pixels")
    p.add_argument("--background", default=DEFAULT_BG,
                   help="Background colour")
    p.add_argument("--make-video", action="store_true",
                   help="Stitch all frames into an MP4")
    p.add_argument("--video-name", default="dendrite.mp4",
                   help="Output video filename (relative to --output-dir)")
    p.add_argument("--fps", type=int, default=12,
                   help="Video frame rate")
    p.add_argument("--limit", type=int, default=0,
                   help="Only render the first N frames (0 = all)")
    p.add_argument("--prescan-limit", type=int, default=0,
                   help="Max frames to scan for global colour range (0 = all)")
    p.add_argument("--no-xvfb", action="store_true",
                   help="Do NOT start Xvfb automatically "
                        "(use if an external Xvfb is already running)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-frame progress lines")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # ── Headless bootstrap ───────────────────────────────────────────────
    pv.OFF_SCREEN = True
    if not args.no_xvfb:
        try:
            pv.start_xvfb(wait=0.2)
        except Exception as exc:  # pragma: no cover
            sys.stderr.write(
                f"WARN: pv.start_xvfb() failed ({exc}). "
                "Assuming an Xvfb is already running or the host is not headless.\n"
            )

    # ── Discover frames ──────────────────────────────────────────────────
    in_dir = args.input_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not in_dir.is_dir():
        sys.stderr.write(f"ERROR: input directory does not exist: {in_dir}\n")
        return 2

    frames = discover_frames(in_dir, args.pattern)
    if not frames:
        sys.stderr.write(
            f"ERROR: no snapshots matching '{args.pattern}' in {in_dir}\n"
            "Run a simulation with output.format=vts first, e.g. `make cuda-run-vtk`.\n"
        )
        return 3

    if args.limit > 0:
        frames = frames[: args.limit]

    sys.stdout.write(f"==> Found {len(frames)} snapshots in {in_dir}\n")
    sys.stdout.write(f"==> Rendering to {out_dir} (view={args.view})\n")

    # ── Prescan global colour ranges ─────────────────────────────────────
    scan_max = args.prescan_limit or None
    t0 = time.perf_counter()
    phi_clim, u_clim = compute_global_ranges(frames, scan_max)
    dt = time.perf_counter() - t0
    sys.stdout.write(
        f"==> Colour range prescan ({dt:.2f}s): "
        f"phi in [{phi_clim[0]:.3f}, {phi_clim[1]:.3f}], "
        f"u in [{u_clim[0]:.3f}, {u_clim[1]:.3f}]\n"
    )

    # ── Render loop ──────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    png_paths: List[Path] = []
    ok = 0
    for i, f in enumerate(frames):
        step = natural_step(str(f))
        png = out_dir / f"frame_{step:06d}.png"
        t_frame = time.perf_counter()
        success = render_frame(
            frame_path=f,
            png_path=png,
            view=args.view,
            phi_clim=phi_clim,
            u_clim=u_clim,
            phi_cmap=args.phi_cmap,
            u_cmap=args.u_cmap,
            iso_value=args.iso_value,
            window_size=tuple(args.window_size),
            background=args.background,
            total=len(frames),
            index=i,
        )
        if success:
            ok += 1
            png_paths.append(png)
            if not args.quiet:
                dt = time.perf_counter() - t_frame
                sys.stdout.write(
                    f"    [{i + 1:>4}/{len(frames)}] step={step:<7} "
                    f"-> {png.name}  ({dt:.2f}s)\n"
                )
                sys.stdout.flush()

    sys.stdout.write(f"==> Rendered {ok}/{len(frames)} frames\n")

    # ── Optional video stitching ─────────────────────────────────────────
    if args.make_video and png_paths:
        stitch_video(png_paths, out_dir / args.video_name, args.fps)

    return 0 if ok > 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
