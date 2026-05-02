#!/usr/bin/env python3
# ============================================================================
# Allen-Cahn-CUDA — production-grade dendrite visualization
# ----------------------------------------------------------------------------
# Reads vtkStructuredGrid (.vts) snapshots produced by the simulation and
# renders publication-quality PNG frames + an optional MP4. Designed for
# headless servers (Lightning.ai, CI, Docker) via PyVista + Xvfb.
#
# Layouts
#   single   one large 3D view: iso surface (phi=0) + opaque cutaway slices
#            on the three back walls of the bounding box. No alpha-blended
#            slice planes ⇒ no moiré speckle.
#   panels   1920×1080 composite: left half is the 3D cutaway, right half
#            is two slice panels (phi mid-z, u mid-z) and the bottom strip
#            is a time-series chart of solid fraction with a moving cursor.
#
# Anti-speckle treatment (vs the naive contour+slice_orthogonal recipe)
#   1. Three opaque back-wall slices instead of three semi-transparent
#      mid-plane slices ⇒ no alpha-blending of overlapping translucent
#      structured grids.
#   2. Taubin smoothing of the marching-cubes mesh ⇒ vertex normals are
#      no longer grid-aligned, killing the "dotted highlight" pattern.
#   3. SSAA (super-sample) + 8× MSAA + low specular on the iso ⇒ no
#      grid-frequency specular aliasing.
#   4. Silhouette outline ⇒ the dendrite reads cleanly against any
#      background colour.
#
# Saturation guard
#   --skip-saturated : skip frames where the solid has reached the box wall
#                       (phi > -0.5 anywhere on the boundary slab). These
#                       frames carry no useful dendrite morphology.
#   without --skip-saturated, saturated frames are rendered with a red
#   "SATURATED" badge in the upper-right.
#
# Usage (standalone)
#   python scripts/visualize_dendrite.py \
#       --input-dir ./out --output-dir ./viz \
#       --layout panels --make-video --fps 12
#
# Usage (Makefile)
#   make cuda-visualize                          # panels (default)
#   make cuda-visualize VIZ_LAYOUT=single
#   make cuda-visualize VIZ_EXTRA_ARGS=--skip-saturated
#
# Self-test
#   python scripts/visualize_dendrite.py --self-test
#       Synthesizes a tiny 24³ grid in /tmp, renders one frame, asserts
#       the PNG is non-empty. Used by tests/visualize_smoke.py and CI.
#
# Dependencies (all in docker/Dockerfile.cuda-dev)
#   pyvista>=0.44,<0.45   numpy>=1.26   imageio[ffmpeg]>=2.34   pillow
#   matplotlib (transitive via pyvista)
#   xvfb (system)         libgl1, libosmesa6 (system)
# ============================================================================

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gc
import numpy as np

# Suppress noisy VTK deprecation/info messages that PyVista cannot silence.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyvista")
warnings.filterwarnings("ignore", category=UserWarning, module="pyvista")

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: pyvista is required. Install with:\n"
        "  pip install 'pyvista>=0.44,<0.45' 'imageio[ffmpeg]>=2.34' pillow\n"
    )
    raise SystemExit(1) from exc

try:
    import matplotlib

    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: matplotlib is required (pulled in by pyvista). "
        "Install with: pip install matplotlib\n"
    )
    raise SystemExit(1) from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: Pillow is required for compositing. Install with: pip install pillow\n"
    )
    raise SystemExit(1) from exc


# ── Constants ────────────────────────────────────────────────────────────────

STEP_RE = re.compile(r"output_(\d+)\.vts$")
DEFAULT_ISO_VALUE = 0.0          # phi=0 is the solid-liquid interface
DEFAULT_LAYOUT = "panels"        # production default — multi-panel
DEFAULT_PHI_CMAP = "coolwarm"    # diverging, perceptually decent
DEFAULT_U_CMAP = "plasma"        # sequential, perceptually-uniform
DEFAULT_BG_BOTTOM = (0.95, 0.95, 0.97)  # near-white
DEFAULT_BG_TOP = (0.65, 0.70, 0.78)     # cool steel-blue
DEFAULT_WINDOW = (1920, 1080)
DEFAULT_SMOOTH_ITERS = 20
DEFAULT_PASS_BAND = 0.10
DEFAULT_SATURATION_THRESHOLD = -0.5  # phi above this on a boundary cell ⇒ wall touched

PANEL_RATIO = 0.62                # main 3D view occupies 62% width
SIDEBAR_HEIGHT_RATIO = 0.18       # time-series sidebar occupies bottom 18%

WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)


# ── Frame discovery and step parsing ─────────────────────────────────────────

def natural_step(path: str) -> int:
    """Extract numeric step from 'output_<step>.vts' or return -1."""
    m = STEP_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def discover_frames(input_dir: Path, pattern: str) -> List[Path]:
    """Return sorted list of snapshot Paths whose name matches the regex."""
    matches = sorted(glob.glob(str(input_dir / pattern)), key=natural_step)
    return [Path(p) for p in matches if natural_step(p) >= 0]


# ── Saturation detection ─────────────────────────────────────────────────────

def detect_saturation(grid: pv.StructuredGrid,
                      threshold: float = DEFAULT_SATURATION_THRESHOLD) -> bool:
    """True if any boundary cell has phi above `threshold`.

    Uses extract_subset to grab the six 1-cell-thick boundary slabs and checks
    the maximum phi value. This is O(N²), not O(N³).
    """
    if "phi" not in grid.point_data:
        return False
    nx, ny, nz = grid.dimensions
    # extract_subset uses inclusive (i_min, i_max, j_min, j_max, k_min, k_max)
    slabs = [
        (0, 0, 0, ny - 1, 0, nz - 1),                # x_min
        (nx - 1, nx - 1, 0, ny - 1, 0, nz - 1),       # x_max
        (0, nx - 1, 0, 0, 0, nz - 1),                 # y_min
        (0, nx - 1, ny - 1, ny - 1, 0, nz - 1),        # y_max
        (0, nx - 1, 0, ny - 1, 0, 0),                 # z_min
        (0, nx - 1, 0, ny - 1, nz - 1, nz - 1),        # z_max
    ]
    for sl in slabs:
        try:
            sub = grid.extract_subset(sl)
        except Exception:
            continue
        if "phi" not in sub.point_data:
            continue
        if float(np.asarray(sub.point_data["phi"]).max()) > threshold:
            return True
    return False


# ── Global prescan (colour ranges + time series) ─────────────────────────────

@dataclass
class ScanResult:
    """Aggregated statistics across all frames, computed once before rendering."""
    phi_clim: Tuple[float, float] = (-1.0, 1.0)
    u_clim: Tuple[float, float] = (-1.0, 0.0)
    steps: List[int] = field(default_factory=list)
    solid_fraction: List[float] = field(default_factory=list)
    mean_u: List[float] = field(default_factory=list)
    saturated: List[bool] = field(default_factory=list)
    grid_dims: Tuple[int, int, int] = (0, 0, 0)
    grid_bounds: Tuple[float, float, float, float, float, float] = (0,) * 6


def compute_global_scan(frames: Sequence[Path],
                        max_scan: Optional[int] = None,
                        sat_threshold: float = DEFAULT_SATURATION_THRESHOLD,
                        progress: bool = True
                        ) -> ScanResult:
    """One pass over every frame: collect colour limits + per-frame scalars.

    For phi we clamp to [-1, +1] (theoretical bound of the order parameter)
    so the colormap is symmetric. For u we take the true [min, max] across
    all frames with a 5% pad so the extremes don't render as flat colour.
    """
    result = ScanResult()
    u_lo, u_hi = math.inf, -math.inf
    phi_lo, phi_hi = math.inf, -math.inf
    scanned = 0
    subset = list(frames if max_scan is None else frames[:max_scan])
    n_total = len(subset)

    for i, f in enumerate(subset):
        if progress and i % max(1, n_total // 20) == 0:
            sys.stdout.write(f"    prescan [{i + 1}/{n_total}] {f.name}\r")
            sys.stdout.flush()
        try:
            grid = pv.read(str(f))
        except Exception as exc:
            sys.stderr.write(f"\nWARN: prescan failed to read {f}: {exc}\n")
            continue

        step = natural_step(str(f))
        result.steps.append(step)

        if not result.grid_dims[0]:
            result.grid_dims = tuple(int(v) for v in grid.dimensions)  # type: ignore
            result.grid_bounds = tuple(float(v) for v in grid.bounds)  # type: ignore

        if "u" in grid.point_data:
            ua = np.asarray(grid.point_data["u"])
            if not np.all(np.isfinite(ua)):
                n_bad = int(np.count_nonzero(~np.isfinite(ua)))
                sys.stderr.write(
                    f"\nWARN: {f.name} has {n_bad} NaN/Inf values in u — "
                    f"clamping to finite range\n")
                ua = np.where(np.isfinite(ua), ua, 0.0)
            u_lo = min(u_lo, float(ua.min()))
            u_hi = max(u_hi, float(ua.max()))
            result.mean_u.append(float(ua.mean()))
        else:
            result.mean_u.append(0.0)

        if "phi" in grid.point_data:
            pa = np.asarray(grid.point_data["phi"])
            if not np.all(np.isfinite(pa)):
                n_bad = int(np.count_nonzero(~np.isfinite(pa)))
                sys.stderr.write(
                    f"\nWARN: {f.name} has {n_bad} NaN/Inf values in phi — "
                    f"clamping to finite range\n")
                pa = np.where(np.isfinite(pa), pa, 0.0)
            phi_lo = min(phi_lo, float(pa.min()))
            phi_hi = max(phi_hi, float(pa.max()))
            result.solid_fraction.append(float((pa > 0.0).mean()))
        else:
            result.solid_fraction.append(0.0)

        result.saturated.append(detect_saturation(grid, sat_threshold))
        scanned += 1

        del grid
        gc.collect()

    if progress:
        sys.stdout.write(" " * 80 + "\r")
        sys.stdout.flush()

    if scanned == 0:
        raise RuntimeError("No readable frames found during prescan")

    # phi is physically in [-1, 1] — clamp for a stable, symmetric colormap.
    phi_lo = max(-1.0, phi_lo if math.isfinite(phi_lo) else -1.0)
    phi_hi = min(+1.0, phi_hi if math.isfinite(phi_hi) else +1.0)
    if phi_lo == phi_hi:
        phi_lo, phi_hi = -1.0, 1.0
    result.phi_clim = (phi_lo, phi_hi)

    if not (math.isfinite(u_lo) and math.isfinite(u_hi)) or u_lo == u_hi:
        u_lo, u_hi = -1.0, 0.0
    pad = 0.05 * max(1e-8, u_hi - u_lo)
    result.u_clim = (u_lo - pad, u_hi + pad)

    return result


# ── Camera ───────────────────────────────────────────────────────────────────

def set_isometric_camera(plotter: pv.Plotter,
                         bounds: Sequence[float],
                         azimuth: float = 35.0,
                         elevation: float = 25.0,
                         distance_factor: float = 1.85) -> None:
    """Place the camera looking from (+x, +y, +z) toward the centre.

    The defaults give a slight asymmetry (azimuth ≠ 45°, elevation < azimuth)
    so the dendrite reads as 3D rather than as an iso-projection silhouette.
    """
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = bounds
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    cz = 0.5 * (z_lo + z_hi)
    diag = math.sqrt((x_hi - x_lo) ** 2 + (y_hi - y_lo) ** 2 + (z_hi - z_lo) ** 2)
    d = distance_factor * 0.5 * diag

    az = math.radians(azimuth)
    el = math.radians(elevation)
    px = cx + d * math.cos(el) * math.cos(az)
    py = cy + d * math.cos(el) * math.sin(az)
    pz = cz + d * math.sin(el)

    plotter.camera_position = [(px, py, pz), (cx, cy, cz), (0.0, 0.0, 1.0)]


# ── Iso surface and slice geometry ───────────────────────────────────────────

def build_isosurface(grid: pv.StructuredGrid,
                     iso_value: float,
                     smooth_iters: int,
                     pass_band: float) -> Optional[pv.PolyData]:
    """Marching cubes at phi=iso_value, then Taubin-smoothed.

    Returns None if the iso surface is empty (no phi=iso crossing). Taubin
    smoothing is shape-preserving (unlike Laplacian) so the dendrite tips
    stay sharp while grid-aligned vertex noise is removed.
    """
    if "phi" not in grid.point_data:
        return None
    try:
        iso = grid.contour(isosurfaces=[iso_value], scalars="phi")
    except Exception as exc:
        sys.stderr.write(f"WARN: contour failed: {exc}\n")
        return None
    if iso.n_points == 0:
        return None

    if smooth_iters > 0:
        try:
            iso = iso.smooth_taubin(n_iter=smooth_iters,
                                    pass_band=pass_band,
                                    feature_smoothing=False,
                                    boundary_smoothing=True,
                                    non_manifold_smoothing=False,
                                    normalize_coordinates=True)
        except (AttributeError, Exception) as exc:
            sys.stderr.write(f"WARN: smooth_taubin failed ({exc}); "
                             "falling back to Laplacian smoothing\n")
            try:
                iso = iso.smooth(n_iter=smooth_iters, relaxation_factor=0.1,
                                 feature_smoothing=False, boundary_smoothing=True)
            except Exception as exc2:
                sys.stderr.write(f"WARN: smooth fallback failed ({exc2}); "
                                 "rendering raw marching-cubes mesh\n")
    return iso


def build_back_wall_slices(grid: pv.StructuredGrid) -> pv.MultiBlock:
    """Three opaque slices on the back walls of the bounding box.

    Camera looks from (+x, +y, +z), so the "back" walls (away from camera)
    are x=x_min, y=y_min, z=z_min. Slicing exactly at the boundary can return
    an empty mesh because vtkCutter operates on cell intersections; use a
    0.5% inset.
    """
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = grid.bounds
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    cz = 0.5 * (z_lo + z_hi)
    eps_x = 0.005 * (x_hi - x_lo)
    eps_y = 0.005 * (y_hi - y_lo)
    eps_z = 0.005 * (z_hi - z_lo)

    blocks: List[pv.PolyData] = []
    for normal, origin in (
        ((1.0, 0.0, 0.0), (x_lo + eps_x, cy, cz)),
        ((0.0, 1.0, 0.0), (cx, y_lo + eps_y, cz)),
        ((0.0, 0.0, 1.0), (cx, cy, z_lo + eps_z)),
    ):
        try:
            s = grid.slice(normal=normal, origin=origin)
        except Exception as exc:
            sys.stderr.write(f"WARN: back-wall slice failed: {exc}\n")
            continue
        if s.n_points > 0:
            blocks.append(s)

    return pv.MultiBlock(blocks) if blocks else pv.MultiBlock()


# ── Plotter assembly: a single 3D view ────────────────────────────────────────

def make_plotter_3d(window_size: Tuple[int, int],
                    bg_bottom: Tuple[float, float, float],
                    bg_top: Tuple[float, float, float]) -> pv.Plotter:
    """Build an off-screen Plotter with three-light rig + gradient bg + AA."""
    plotter = pv.Plotter(off_screen=True,
                         window_size=list(window_size),
                         lighting="three lights")
    plotter.set_background(color=bg_bottom, top=bg_top)
    # SSAA = supersample (renders at 2× then downsamples; cures most edge aliasing).
    # MSAA = multi-sample (handled by VTK at primitive level; cures triangle edges).
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    try:
        # `multi_samples` is a Plotter render-window setting in PyVista 0.44+.
        plotter.ren_win.SetMultiSamples(8)  # type: ignore[attr-defined]
    except Exception:
        pass
    return plotter


def populate_3d_scene(plotter: pv.Plotter,
                      grid: pv.StructuredGrid,
                      *,
                      iso: Optional[pv.PolyData],
                      slices: pv.MultiBlock,
                      u_clim: Tuple[float, float],
                      phi_clim: Tuple[float, float],
                      u_cmap: str,
                      phi_cmap: str,
                      add_silhouette: bool,
                      show_scalar_bar: bool,
                      title: str) -> None:
    # Back-wall slices, opaque, lit by ambient only (so flat colour reads cleanly)
    if slices.n_blocks > 0:
        for block in slices:
            if block is None:
                continue
            plotter.add_mesh(
                block,
                scalars="phi",
                cmap=phi_cmap,
                clim=phi_clim,
                opacity=1.0,
                lighting=False,           # flat shaded; the slice itself is the data
                show_scalar_bar=False,
                interpolate_before_map=True,
                name=f"back_wall_{id(block)}",
            )

    # Iso surface, smooth-shaded, low specular, coloured by u
    if iso is not None and iso.n_points > 0:
        scalars_kwargs = {}
        if "u" in iso.point_data:
            scalars_kwargs = dict(scalars="u", cmap=u_cmap, clim=u_clim)
        else:
            scalars_kwargs = dict(color="lightsteelblue")

        plotter.add_mesh(
            iso,
            smooth_shading=True,
            ambient=0.30,
            diffuse=0.75,
            specular=0.15,        # low — was 0.45 in the old script
            specular_power=8,     # softer highlights
            interpolate_before_map=True,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={
                "title": "u (temperature)",
                "title_font_size": 16,
                "label_font_size": 13,
                "n_labels": 5,
                "position_x": 0.84,
                "position_y": 0.10,
                "width": 0.10,
                "height": 0.55,
                "color": "black",
                "fmt": "%.2f",
            } if show_scalar_bar else {},
            name="dendrite",
            **scalars_kwargs,
        )

        if add_silhouette:
            try:
                plotter.add_silhouette(
                    iso,
                    color="black",
                    line_width=2.0,
                    decimate=0.5,
                )
            except Exception as exc:
                sys.stderr.write(f"WARN: add_silhouette failed: {exc}\n")

    # Bounding box outline (thin)
    try:
        plotter.add_mesh(grid.outline(), color="black", line_width=1.2,
                         name="bbox", lighting=False)
    except Exception:
        pass

    # Axes gizmo (lower-left)
    plotter.add_axes(interactive=False, line_width=3, labels_off=False)

    # Title overlay (upper-left)
    plotter.add_text(title,
                     position="upper_left",
                     font_size=12,
                     color="black",
                     shadow=False,
                     name="title")


# ── Slice panels (small 2D-ish views colored by phi or u) ─────────────────────

def render_slice_panel(grid: pv.StructuredGrid,
                       scalar: str,
                       cmap: str,
                       clim: Tuple[float, float],
                       title: str,
                       window_size: Tuple[int, int],
                       bg: Tuple[float, float, float]) -> np.ndarray:
    """Render a single mid-z slice as an RGBA numpy array."""
    if scalar not in grid.point_data:
        return np.full((window_size[1], window_size[0], 4), 200, dtype=np.uint8)

    cz = 0.5 * (grid.bounds[4] + grid.bounds[5])
    eps = 0.005 * (grid.bounds[5] - grid.bounds[4])
    try:
        s = grid.slice(normal=(0.0, 0.0, 1.0),
                       origin=(0.0, 0.0, cz + eps))
    except Exception as exc:
        sys.stderr.write(f"WARN: slice panel ({scalar}) failed: {exc}\n")
        return np.full((window_size[1], window_size[0], 4), 200, dtype=np.uint8)
    if s.n_points == 0:
        return np.full((window_size[1], window_size[0], 4), 200, dtype=np.uint8)

    p = pv.Plotter(off_screen=True, window_size=list(window_size))
    p.set_background(color=bg)
    try:
        p.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    p.add_mesh(s, scalars=scalar, cmap=cmap, clim=clim,
               show_scalar_bar=True,
               interpolate_before_map=True,
               lighting=False,
               scalar_bar_args={
                   "title": scalar,
                   "title_font_size": 14,
                   "label_font_size": 11,
                   "n_labels": 3,
                   "position_x": 0.83,
                   "position_y": 0.10,
                   "width": 0.08,
                   "height": 0.50,
                   "color": "black",
                   "fmt": "%.2f",
               })
    p.add_text(title, position="upper_left", font_size=12,
               color="black", shadow=False)
    p.add_mesh(grid.outline(), color="black", line_width=1.0, lighting=False)
    p.view_xy()
    p.camera.zoom(1.18)

    rgb = p.screenshot(transparent_background=False, return_img=True)
    p.close()
    if rgb is None:
        return np.full((window_size[1], window_size[0], 4), 200, dtype=np.uint8)
    if rgb.shape[2] == 3:
        alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
        rgb = np.concatenate([rgb, alpha], axis=2)
    return rgb


# ── Time-series sidebar (matplotlib) ─────────────────────────────────────────

def render_timeseries_sidebar(scan: ScanResult,
                              current_step: int,
                              size_px: Tuple[int, int],
                              dpi: int = 110) -> np.ndarray:
    """Render the solid-fraction-vs-step curve with a moving cursor.

    Returns RGBA numpy array of shape (size_px[1], size_px[0], 4).
    """
    fig_w_in = size_px[0] / dpi
    fig_h_in = size_px[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    fig.patch.set_facecolor("white")

    if scan.steps:
        steps = np.asarray(scan.steps, dtype=float)
        sf = np.asarray(scan.solid_fraction, dtype=float)
        ax.plot(steps, sf, color="#2c3e8a", linewidth=2.0, label="solid fraction")
        ax.fill_between(steps, 0.0, sf, color="#2c3e8a", alpha=0.18)

        if scan.mean_u:
            ax2 = ax.twinx()
            mu = np.asarray(scan.mean_u, dtype=float)
            ax2.plot(steps, mu, color="#c64a3b", linewidth=1.6,
                     linestyle="--", label="⟨u⟩")
            ax2.set_ylabel("⟨u⟩", color="#c64a3b", fontsize=10)
            ax2.tick_params(axis="y", labelcolor="#c64a3b", labelsize=9)
            for spine in ("top",):
                ax2.spines[spine].set_visible(False)

        # Saturation shading
        sat_mask = np.asarray(scan.saturated, dtype=bool)
        if sat_mask.any():
            sat_steps = steps[sat_mask]
            ax.axvspan(sat_steps.min(), steps.max(), color="#ffd2cc",
                       alpha=0.55, zorder=0,
                       label="saturated (wall reached)")

        # Cursor
        ax.axvline(current_step, color="black", linewidth=1.5, alpha=0.7)
        ax.scatter([current_step],
                   [_interp(steps, sf, current_step)],
                   s=40, color="black", zorder=5)

        x_lo, x_hi = float(steps.min()), float(steps.max())
        if x_lo == x_hi:
            # Single-frame edge case — pad so matplotlib doesn't complain about a
            # singular transform.
            pad = max(1.0, abs(x_lo) * 0.05)
            x_lo -= pad
            x_hi += pad
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, max(0.05, float(sf.max()) * 1.08))
        ax.set_xlabel("simulation step", fontsize=10)
        ax.set_ylabel("solid fraction", color="#2c3e8a", fontsize=10)
        ax.tick_params(axis="y", labelcolor="#2c3e8a", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        for spine in ("top",):
            ax.spines[spine].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    else:
        ax.text(0.5, 0.5, "no time-series data",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(pad=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    img = img.resize(size_px, Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _interp(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    if xs.size == 0:
        return 0.0
    i = int(np.searchsorted(xs, x))
    if i <= 0:
        return float(ys[0])
    if i >= xs.size:
        return float(ys[-1])
    x0, x1 = float(xs[i - 1]), float(xs[i])
    y0, y1 = float(ys[i - 1]), float(ys[i])
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


# ── 3D scene render ──────────────────────────────────────────────────────────

def render_3d_view(grid: pv.StructuredGrid,
                   *,
                   iso_value: float,
                   smooth_iters: int,
                   pass_band: float,
                   u_clim: Tuple[float, float],
                   phi_clim: Tuple[float, float],
                   u_cmap: str,
                   phi_cmap: str,
                   window_size: Tuple[int, int],
                   bg_bottom: Tuple[float, float, float],
                   bg_top: Tuple[float, float, float],
                   add_silhouette_outline: bool,
                   show_scalar_bar: bool,
                   title: str) -> np.ndarray:
    """Render the main cutaway 3D view to an RGBA numpy array."""
    iso = build_isosurface(grid, iso_value, smooth_iters, pass_band)
    slices = build_back_wall_slices(grid)

    plotter = make_plotter_3d(window_size, bg_bottom, bg_top)
    populate_3d_scene(plotter, grid,
                      iso=iso, slices=slices,
                      u_clim=u_clim, phi_clim=phi_clim,
                      u_cmap=u_cmap, phi_cmap=phi_cmap,
                      add_silhouette=add_silhouette_outline,
                      show_scalar_bar=show_scalar_bar,
                      title=title)
    set_isometric_camera(plotter, grid.bounds)

    rgb = plotter.screenshot(transparent_background=False, return_img=True)
    plotter.close()

    if rgb is None:
        return np.full((window_size[1], window_size[0], 4), 245, dtype=np.uint8)
    if rgb.shape[2] == 3:
        alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
        rgb = np.concatenate([rgb, alpha], axis=2)
    return rgb


# ── Compositing ──────────────────────────────────────────────────────────────

def overlay_saturation_badge(img: Image.Image) -> Image.Image:
    """Burn a red 'SATURATED — wall reached' banner into the upper-right corner."""
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    text = "SATURATED — wall reached"
    font = _load_font(20)
    if font is not None:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = 8 * len(text), 18
    pad = 12
    x = out.width - tw - 3 * pad
    y = pad
    draw.rounded_rectangle((x, y, x + tw + 2 * pad, y + th + 2 * pad),
                           radius=8, fill=(204, 0, 0, 220))
    draw.text((x + pad, y + pad), text, fill=(255, 255, 255, 255), font=font)
    return out


def _load_font(size: int) -> Optional[ImageFont.ImageFont]:
    """Best-effort load of a truetype font; falls back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size=size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def compose_panels_layout(main_3d: np.ndarray,
                          phi_panel: np.ndarray,
                          u_panel: np.ndarray,
                          sidebar: np.ndarray,
                          final_size: Tuple[int, int],
                          saturated: bool) -> Image.Image:
    """Compose the 'panels' layout: big 3D + 2 small slices + bottom time-series."""
    W, H = final_size
    # Geometry
    sidebar_h = int(round(SIDEBAR_HEIGHT_RATIO * H))
    main_w = int(round(PANEL_RATIO * W))
    side_w = W - main_w
    main_h = H - sidebar_h
    panel_h = main_h // 2

    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    # Helpers
    def _paste(arr: np.ndarray, dst_size: Tuple[int, int], at: Tuple[int, int]) -> None:
        if arr.size == 0:
            return
        im = Image.fromarray(arr, mode="RGBA")
        if im.size != dst_size:
            im = im.resize(dst_size, Image.LANCZOS)
        canvas.paste(im, at, im)

    _paste(main_3d, (main_w, main_h), (0, 0))
    _paste(phi_panel, (side_w, panel_h), (main_w, 0))
    _paste(u_panel,   (side_w, main_h - panel_h), (main_w, panel_h))
    _paste(sidebar, (W, sidebar_h), (0, main_h))

    # Subtle separators
    sep = ImageDraw.Draw(canvas)
    sep.line([(main_w, 0), (main_w, main_h)], fill=(180, 180, 180, 255), width=1)
    sep.line([(main_w, panel_h), (W, panel_h)], fill=(180, 180, 180, 255), width=1)
    sep.line([(0, main_h), (W, main_h)], fill=(180, 180, 180, 255), width=1)

    if saturated:
        canvas = overlay_saturation_badge(canvas)
    return canvas


def compose_single_layout(main_3d: np.ndarray,
                          final_size: Tuple[int, int],
                          saturated: bool) -> Image.Image:
    W, H = final_size
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    if main_3d.size > 0:
        im = Image.fromarray(main_3d, mode="RGBA")
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        canvas.paste(im, (0, 0), im)
    if saturated:
        canvas = overlay_saturation_badge(canvas)
    return canvas


# ── Per-frame orchestration ──────────────────────────────────────────────────

@dataclass
class RenderConfig:
    layout: str
    iso_value: float
    smooth_iters: int
    pass_band: float
    phi_cmap: str
    u_cmap: str
    window_size: Tuple[int, int]
    bg_bottom: Tuple[float, float, float]
    bg_top: Tuple[float, float, float]
    silhouette: bool
    skip_saturated: bool


def render_frame(frame_path: Path,
                 png_path: Path,
                 scan: ScanResult,
                 cfg: RenderConfig,
                 index: int,
                 total: int) -> bool:
    """Render one .vts snapshot to PNG. Returns True on success / non-skip."""
    try:
        grid = pv.read(str(frame_path))
    except Exception as exc:
        sys.stderr.write(f"ERROR reading {frame_path}: {exc}\n")
        return False

    for field_name in ("phi", "u"):
        if field_name in grid.point_data:
            arr = np.asarray(grid.point_data[field_name])
            if not np.all(np.isfinite(arr)):
                n_bad = int(np.count_nonzero(~np.isfinite(arr)))
                sys.stderr.write(
                    f"WARN: {frame_path.name} has {n_bad} NaN/Inf in {field_name} "
                    f"— replacing with 0 for rendering\n")
                grid.point_data[field_name] = np.where(np.isfinite(arr), arr, 0.0)

    step = natural_step(str(frame_path))
    saturated = detect_saturation(grid)
    if saturated and cfg.skip_saturated:
        sys.stdout.write(f"    [{index + 1:>4}/{total}] step={step:<7} "
                         f"-> SKIPPED (saturated)\n")
        return False

    title = (f"Allen-Cahn dendrite\n"
             f"step = {step}   frame {index + 1}/{total}\n"
             f"grid {grid.dimensions[0]}×{grid.dimensions[1]}×{grid.dimensions[2]}")

    W, H = cfg.window_size

    if cfg.layout == "single":
        main = render_3d_view(
            grid,
            iso_value=cfg.iso_value,
            smooth_iters=cfg.smooth_iters,
            pass_band=cfg.pass_band,
            u_clim=scan.u_clim,
            phi_clim=scan.phi_clim,
            u_cmap=cfg.u_cmap,
            phi_cmap=cfg.phi_cmap,
            window_size=(W, H),
            bg_bottom=cfg.bg_bottom,
            bg_top=cfg.bg_top,
            add_silhouette_outline=cfg.silhouette,
            show_scalar_bar=True,
            title=title,
        )
        out = compose_single_layout(main, (W, H), saturated)
    else:  # panels
        sidebar_h = int(round(SIDEBAR_HEIGHT_RATIO * H))
        main_w = int(round(PANEL_RATIO * W))
        side_w = W - main_w
        main_h = H - sidebar_h
        panel_h = main_h // 2
        panel_h_bot = main_h - panel_h

        main = render_3d_view(
            grid,
            iso_value=cfg.iso_value,
            smooth_iters=cfg.smooth_iters,
            pass_band=cfg.pass_band,
            u_clim=scan.u_clim,
            phi_clim=scan.phi_clim,
            u_cmap=cfg.u_cmap,
            phi_cmap=cfg.phi_cmap,
            window_size=(main_w, main_h),
            bg_bottom=cfg.bg_bottom,
            bg_top=cfg.bg_top,
            add_silhouette_outline=cfg.silhouette,
            show_scalar_bar=True,
            title=title,
        )
        phi_panel = render_slice_panel(
            grid, scalar="phi", cmap=cfg.phi_cmap, clim=scan.phi_clim,
            title="phi (mid-z slice)",
            window_size=(side_w, panel_h),
            bg=cfg.bg_bottom,
        )
        u_panel = render_slice_panel(
            grid, scalar="u", cmap=cfg.u_cmap, clim=scan.u_clim,
            title="u (mid-z slice)",
            window_size=(side_w, panel_h_bot),
            bg=cfg.bg_bottom,
        )
        sidebar = render_timeseries_sidebar(scan, step, (W, sidebar_h))

        out = compose_panels_layout(main, phi_panel, u_panel, sidebar,
                                    (W, H), saturated)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.convert("RGB").save(str(png_path), format="PNG", optimize=True)
    except Exception as exc:
        sys.stderr.write(f"ERROR writing {png_path}: {exc}\n")
        return False
    finally:
        del grid
        gc.collect()
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


# ── Self-test (used by tests/visualize_smoke.py and CI) ──────────────────────

def run_self_test(workdir: Optional[Path] = None) -> int:
    """Synthesize a tiny 24³ structured grid + run a single-frame render.

    This exercises the full pipeline (pyvista read, contour, smoothing,
    cutaway slices, composition, PNG write) on data we generate ourselves,
    so a CI runner with no GPU and no real simulation output can still
    catch viz-side regressions.
    """
    sys.stdout.write("==> visualize_dendrite self-test\n")
    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb(wait=0.2)
    except Exception as exc:
        sys.stderr.write(f"WARN: pv.start_xvfb() failed ({exc}); "
                         "continuing — assume an X server is available.\n")

    tmpdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ac_viz_"))
    in_dir = tmpdir / "out"
    out_dir = tmpdir / "viz"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 24
    dx = 0.5
    grid = pv.StructuredGrid()
    xs = np.arange(n) * dx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.column_stack([X.ravel(order="F"),
                           Y.ravel(order="F"),
                           Z.ravel(order="F")])
    grid.points = pts
    grid.dimensions = [n, n, n]

    cx = cy = cz = 0.5 * (n - 1) * dx
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    r0 = 2.5
    phi = -np.tanh((r - r0) / 1.0)        # -1 outside, +1 inside, smooth interface
    u = -0.6 * np.exp(-((r - r0) / 3.0) ** 2)  # warm core, cool far-field

    grid.point_data["phi"] = phi.ravel(order="F")
    grid.point_data["u"] = u.ravel(order="F")

    snapshot = in_dir / "output_0.vts"
    grid.save(str(snapshot))
    sys.stdout.write(f"    wrote synthetic {snapshot} ({n}^3)\n")

    frames = discover_frames(in_dir, "output_*.vts")
    if not frames:
        sys.stderr.write("ERROR: discover_frames returned nothing\n")
        return 1

    scan = compute_global_scan(frames, progress=False)
    sys.stdout.write(
        f"    prescan ok: phi in [{scan.phi_clim[0]:.3f}, {scan.phi_clim[1]:.3f}], "
        f"u in [{scan.u_clim[0]:.3f}, {scan.u_clim[1]:.3f}]\n"
    )

    cfg = RenderConfig(
        layout="panels",
        iso_value=0.0,
        smooth_iters=DEFAULT_SMOOTH_ITERS,
        pass_band=DEFAULT_PASS_BAND,
        phi_cmap=DEFAULT_PHI_CMAP,
        u_cmap=DEFAULT_U_CMAP,
        window_size=(960, 540),    # smaller, just for smoke
        bg_bottom=DEFAULT_BG_BOTTOM,
        bg_top=DEFAULT_BG_TOP,
        silhouette=True,
        skip_saturated=False,
    )
    png = out_dir / "frame_000000.png"
    ok = render_frame(frames[0], png, scan, cfg, index=0, total=1)
    if not ok or not png.exists() or png.stat().st_size < 5_000:
        sys.stderr.write(
            f"ERROR: self-test render failed (ok={ok}, "
            f"exists={png.exists()}, "
            f"size={png.stat().st_size if png.exists() else 0})\n"
        )
        return 2

    sys.stdout.write(f"    render ok: {png} ({png.stat().st_size} bytes)\n")
    sys.stdout.write("==> self-test PASSED\n")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="visualize_dendrite",
        description=(
            "Render Allen-Cahn .vts snapshots to publication-quality PNGs "
            "(and optionally MP4)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, default=Path("out"),
                   help="Directory containing output_*.vts files")
    p.add_argument("--output-dir", type=Path, default=Path("viz"),
                   help="Directory to write frame_*.png (and optional mp4)")
    p.add_argument("--pattern", default="output_*.vts",
                   help="Glob pattern for snapshot files")

    # Layout
    p.add_argument("--layout", choices=("panels", "single"), default=DEFAULT_LAYOUT,
                   help="panels = 3D + slice panels + time-series sidebar; "
                        "single = just the 3D cutaway view")
    # Backwards compatibility with the old --view flag
    p.add_argument("--view", choices=("iso", "slice", "combined"),
                   default=None,
                   help="(deprecated) old viewing mode; if 'iso' or 'combined', "
                        "uses --layout single, otherwise --layout panels")

    # Iso surface
    p.add_argument("--iso-value", type=float, default=DEFAULT_ISO_VALUE,
                   help="phi isosurface value (solid-liquid interface)")
    p.add_argument("--smooth-iters", type=int, default=DEFAULT_SMOOTH_ITERS,
                   help="Taubin smoothing iterations for the iso mesh "
                        "(0 disables smoothing)")
    p.add_argument("--pass-band", type=float, default=DEFAULT_PASS_BAND,
                   help="Taubin pass-band frequency (lower = smoother)")
    p.add_argument("--no-silhouette", action="store_true",
                   help="Disable the black silhouette outline on the iso surface")

    # Colours
    p.add_argument("--phi-cmap", default=DEFAULT_PHI_CMAP,
                   help="Colormap for phi (slices and back walls)")
    p.add_argument("--u-cmap", default=DEFAULT_U_CMAP,
                   help="Colormap for u (iso surface and u panel)")
    p.add_argument("--bg-bottom", default=",".join(f"{c:.3f}" for c in DEFAULT_BG_BOTTOM),
                   help="Bottom background colour 'r,g,b' (0..1)")
    p.add_argument("--bg-top", default=",".join(f"{c:.3f}" for c in DEFAULT_BG_TOP),
                   help="Top background colour 'r,g,b' (0..1)")

    # Window
    p.add_argument("--window-size", nargs=2, type=int, metavar=("W", "H"),
                   default=list(DEFAULT_WINDOW),
                   help="Output canvas size in pixels")

    # Saturation
    p.add_argument("--skip-saturated", action="store_true",
                   help="Drop frames where the solid has reached the box wall")
    p.add_argument("--saturation-threshold", type=float,
                   default=DEFAULT_SATURATION_THRESHOLD,
                   help="phi value above which a boundary cell is 'saturated'")

    # Video
    p.add_argument("--make-video", action="store_true",
                   help="Stitch all frames into an MP4")
    p.add_argument("--video-name", default="dendrite.mp4",
                   help="Output video filename (relative to --output-dir)")
    p.add_argument("--fps", type=int, default=12,
                   help="Video frame rate")

    # Limits
    p.add_argument("--limit", type=int, default=0,
                   help="Only render the first N frames (0 = all)")
    p.add_argument("--prescan-limit", type=int, default=0,
                   help="Max frames to scan for global colour range / time-series "
                        "(0 = all frames)")

    # Bootstrap
    p.add_argument("--no-xvfb", action="store_true",
                   help="Do NOT start Xvfb (use if an external Xvfb / X is running)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-frame progress lines")
    p.add_argument("--self-test", action="store_true",
                   help="Run an internal smoke test on synthetic data and exit")

    # Stats dump (useful for CI / debugging)
    p.add_argument("--scan-json", type=Path, default=None,
                   help="If given, write the prescan ScanResult to this JSON path")

    return p.parse_args(argv)


def _parse_color_triplet(text: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 'r,g,b' got {text!r}")
    r, g, b = (float(p) for p in parts)
    return (r, g, b)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

    # ── Headless bootstrap ───────────────────────────────────────────────
    pv.OFF_SCREEN = True
    if not args.no_xvfb:
        try:
            pv.start_xvfb(wait=0.2)
        except Exception as exc:  # pragma: no cover
            sys.stderr.write(
                f"WARN: pv.start_xvfb() failed ({exc}). Assuming an Xvfb is "
                "already running or the host is not headless.\n"
            )

    # ── Resolve layout ───────────────────────────────────────────────────
    layout = args.layout
    if args.view is not None:
        # Legacy compat
        if args.view in ("iso", "combined"):
            layout = "single"
        elif args.view == "slice":
            layout = "panels"

    try:
        bg_bottom = _parse_color_triplet(args.bg_bottom)
        bg_top = _parse_color_triplet(args.bg_top)
    except ValueError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2

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
            "Run a simulation with output.format=vts first, e.g. "
            "`make cuda-run-vtk` or `make cuda-run-dendrite`.\n"
        )
        return 3

    if args.limit > 0:
        frames = frames[: args.limit]

    sys.stdout.write(f"==> Found {len(frames)} snapshots in {in_dir}\n")
    sys.stdout.write(f"==> Layout = {layout}, output = {out_dir}\n")

    # ── Prescan ──────────────────────────────────────────────────────────
    scan_max = args.prescan_limit or None
    t0 = time.perf_counter()
    scan = compute_global_scan(frames, scan_max,
                               sat_threshold=args.saturation_threshold,
                               progress=not args.quiet)
    dt = time.perf_counter() - t0
    sys.stdout.write(
        f"==> Prescan ({dt:.2f}s): "
        f"phi in [{scan.phi_clim[0]:.3f}, {scan.phi_clim[1]:.3f}], "
        f"u in [{scan.u_clim[0]:.3f}, {scan.u_clim[1]:.3f}], "
        f"saturated frames = {sum(scan.saturated)}/{len(scan.saturated)}\n"
    )

    if args.scan_json is not None:
        args.scan_json.parent.mkdir(parents=True, exist_ok=True)
        with args.scan_json.open("w") as fh:
            json.dump({
                "phi_clim": list(scan.phi_clim),
                "u_clim": list(scan.u_clim),
                "steps": scan.steps,
                "solid_fraction": scan.solid_fraction,
                "mean_u": scan.mean_u,
                "saturated": scan.saturated,
                "grid_dims": list(scan.grid_dims),
                "grid_bounds": list(scan.grid_bounds),
            }, fh, indent=2)
        sys.stdout.write(f"==> Wrote prescan JSON: {args.scan_json}\n")

    # ── Render ───────────────────────────────────────────────────────────
    cfg = RenderConfig(
        layout=layout,
        iso_value=args.iso_value,
        smooth_iters=max(0, args.smooth_iters),
        pass_band=args.pass_band,
        phi_cmap=args.phi_cmap,
        u_cmap=args.u_cmap,
        window_size=tuple(args.window_size),  # type: ignore[arg-type]
        bg_bottom=bg_bottom,
        bg_top=bg_top,
        silhouette=not args.no_silhouette,
        skip_saturated=args.skip_saturated,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_paths: List[Path] = []
    ok = 0
    for i, f in enumerate(frames):
        step = natural_step(str(f))
        png = out_dir / f"frame_{step:06d}.png"
        t_frame = time.perf_counter()
        success = render_frame(f, png, scan, cfg, index=i, total=len(frames))
        if success:
            ok += 1
            png_paths.append(png)
            if not args.quiet:
                dt_f = time.perf_counter() - t_frame
                sys.stdout.write(
                    f"    [{i + 1:>4}/{len(frames)}] step={step:<7} "
                    f"-> {png.name}  ({dt_f:.2f}s)\n"
                )
                sys.stdout.flush()

    sys.stdout.write(f"==> Rendered {ok}/{len(frames)} frames\n")

    # ── Optional video stitching ─────────────────────────────────────────
    if args.make_video and png_paths:
        stitch_video(png_paths, out_dir / args.video_name, args.fps)

    return 0 if ok > 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
