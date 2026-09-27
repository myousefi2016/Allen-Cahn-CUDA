#!/usr/bin/env python3
# ============================================================================
# Smoke test for scripts/visualize_dendrite.py.
#
# Generates a small synthetic structured grid via PyVista, runs every public
# entry point of the visualizer (prescan, single-frame render in both
# layouts, MP4 stitch, saturation detection, scan-json dump), and asserts:
#
#   1. The CLI's --self-test mode succeeds (exit 0)
#   2. The 'panels' layout produces a non-trivial PNG (>10 KB, RGB image of
#      requested dimensions)
#   3. The 'single' layout produces a non-trivial PNG
#   4. detect_saturation returns True when the seed reaches the wall and
#      False when it does not
#   5. The scan-json dump is well-formed and contains the expected keys;
#      its phi/u data ranges equal the synthetic min/max (distinct from the
#      padded u colour range), and the log prints them labelled as such
#   6. MP4 stitching produces a playable file (positive size, non-empty
#      header that ffprobe could read — we just check size)
#   7. Rendering a second simulation into a reused output dir never keeps
#      or stitches frames rendered from the first simulation
#
# Used by the Makefile target `make cuda-visualize-self-test`. Designed to
# run inside the cuda-dev image with no GPU and no real .vts files.
#
# Exit codes
#   0  all checks passed
#   1  a sub-test failed
#   2  PyVista / numpy / pillow not installed (hard prerequisite)
# ============================================================================

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure scripts/ is on the path so we can import the script as a module.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import numpy as np
    import pyvista as pv  # noqa: F401
    from PIL import Image
    import imageio.v2 as imageio
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        f"ERROR: missing dependency ({exc}). "
        "This smoke test must run inside the cuda-dev image.\n"
    )
    raise SystemExit(2)

import visualize_dendrite as viz  # type: ignore  # noqa: E402


def _synthetic_fields(*, n: int, dx: float, r0: float):
    """Point coordinates (X, Y, Z) and the phi/u fields of the synthetic seed."""
    xs = np.arange(n) * dx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    cx = cy = cz = 0.5 * (n - 1) * dx
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    phi = -np.tanh((r - r0) / 1.0)
    u = -0.6 * np.exp(-((r - r0) / 3.0) ** 2)
    return (X, Y, Z), phi, u


def _make_tmp_grid(tmpdir: Path, *, n: int, dx: float,
                   r0: float, name: str = "output_0.vts") -> Path:
    """Write a synthetic vtkStructuredGrid with phi/u to disk."""
    in_dir = tmpdir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)

    (X, Y, Z), phi, u = _synthetic_fields(n=n, dx=dx, r0=r0)

    grid = pv.StructuredGrid()
    grid.points = np.column_stack([X.ravel(order="F"),
                                   Y.ravel(order="F"),
                                   Z.ravel(order="F")])
    grid.dimensions = [n, n, n]
    grid.point_data["phi"] = phi.ravel(order="F")
    grid.point_data["u"] = u.ravel(order="F")

    out = in_dir / name
    grid.save(str(out))
    return out


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        sys.stderr.write(f"FAIL: {msg}\n")
        raise SystemExit(1)


def _check_png(path: Path, label: str,
               min_bytes: int = 10_000,
               expected_size: tuple = None) -> None:
    _assert(path.exists(), f"{label}: PNG was not created")
    size = path.stat().st_size
    _assert(size >= min_bytes, f"{label}: PNG too small ({size} < {min_bytes})")
    with Image.open(path) as im:
        _assert(im.format == "PNG", f"{label}: not a PNG ({im.format})")
        _assert(im.mode in ("RGB", "RGBA"),
                f"{label}: unexpected mode {im.mode}")
        if expected_size is not None:
            _assert(im.size == expected_size,
                    f"{label}: dim {im.size} != expected {expected_size}")
        # Cheap sanity — confirm at least 16 unique colours (catches blank fills)
        sample = np.asarray(im.convert("RGB"))[::8, ::8].reshape(-1, 3)
        unique = len({tuple(c) for c in sample})
        _assert(unique > 16,
                f"{label}: only {unique} unique colours in PNG sample (expected >16)")
    sys.stdout.write(f"  PASS  {label} -- {path.name} ({size:,} bytes)\n")


def _bootstrap_xvfb() -> None:
    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb(wait=0.3)
    except Exception:
        pass


def main() -> int:
    sys.stdout.write("==> visualize_smoke: starting\n")
    _bootstrap_xvfb()

    tmpdir = Path(tempfile.mkdtemp(prefix="ac_viz_smoke_"))
    sys.stdout.write(f"    tmpdir = {tmpdir}\n")

    try:
        # ── 1. CLI --self-test mode ────────────────────────────────────
        cli_path = ROOT / "scripts" / "visualize_dendrite.py"
        rc = subprocess.call([sys.executable, str(cli_path), "--self-test"])
        _assert(rc == 0, f"--self-test exited rc={rc}")
        sys.stdout.write("  PASS  CLI --self-test\n")

        # ── 2. Discover & prescan + saturation detection (non-saturated) ──
        small = _make_tmp_grid(tmpdir / "small", n=24, dx=0.5, r0=2.5,
                                name="output_0.vts")
        frames = viz.discover_frames(small.parent, "output_*.vts")
        _assert(len(frames) == 1, "discover_frames did not pick up the synthetic file")

        scan = viz.compute_global_scan(frames, progress=False)
        _assert(-1.001 <= scan.phi_clim[0] <= -0.5,
                f"phi_clim low out of range: {scan.phi_clim}")
        _assert(0.5 <= scan.phi_clim[1] <= 1.001,
                f"phi_clim high out of range: {scan.phi_clim}")
        _assert(len(scan.solid_fraction) == 1, "solid_fraction wrong length")
        _assert(0 <= scan.solid_fraction[0] <= 1.0, "solid fraction not in [0,1]")
        _assert(scan.saturated == [False],
                f"non-saturated grid mis-classified: {scan.saturated}")
        # Data ranges must be the synthetic min/max exactly; the u colour
        # range must be the padded one that strictly contains it.
        _, phi_ref, u_ref = _synthetic_fields(n=24, dx=0.5, r0=2.5)
        phi_range = (float(phi_ref.min()), float(phi_ref.max()))
        u_range = (float(u_ref.min()), float(u_ref.max()))
        phi_scanned = getattr(scan, "phi_data_range", None)
        u_scanned = getattr(scan, "u_data_range", None)
        _assert(phi_scanned is not None
                and np.allclose(phi_scanned, phi_range, rtol=0, atol=1e-12),
                f"phi_data_range {phi_scanned} != synthetic {phi_range}")
        _assert(u_scanned is not None
                and np.allclose(u_scanned, u_range, rtol=0, atol=1e-12),
                f"u_data_range {u_scanned} != synthetic {u_range}")
        _assert(scan.u_clim[0] < u_range[0] and scan.u_clim[1] > u_range[1],
                f"u_clim {scan.u_clim} does not pad data range {u_range}")
        sys.stdout.write("  PASS  prescan + non-saturated detection\n")

        # ── 3. Saturation detection on a wall-touching grid ────────────
        sat_path = _make_tmp_grid(tmpdir / "sat", n=24, dx=0.5, r0=12.0,
                                  name="output_0.vts")
        sat_grid = pv.read(str(sat_path))
        _assert(viz.detect_saturation(sat_grid),
                "saturation guard failed on wall-touching grid")
        sys.stdout.write("  PASS  saturated detection on wall-touching grid\n")

        # ── 4. Render: panels layout ───────────────────────────────────
        out_panels = tmpdir / "viz_panels"
        out_panels.mkdir(parents=True, exist_ok=True)
        cfg_p = viz.RenderConfig(
            layout="panels",
            iso_value=0.0,
            smooth_iters=10,
            pass_band=0.10,
            phi_cmap=viz.DEFAULT_PHI_CMAP,
            u_cmap=viz.DEFAULT_U_CMAP,
            window_size=(960, 540),
            bg_bottom=viz.DEFAULT_BG_BOTTOM,
            bg_top=viz.DEFAULT_BG_TOP,
            silhouette=True,
            skip_saturated=False,
        )
        png_p = out_panels / "frame_panels.png"
        ok = viz.render_frame(frames[0], png_p, scan, cfg_p, index=0, total=1)
        _assert(ok, "render_frame (panels) returned False")
        _check_png(png_p, "panels-layout PNG", expected_size=(960, 540))

        # ── 5. Render: single layout ───────────────────────────────────
        out_single = tmpdir / "viz_single"
        out_single.mkdir(parents=True, exist_ok=True)
        cfg_s = viz.RenderConfig(
            layout="single",
            iso_value=0.0,
            smooth_iters=10,
            pass_band=0.10,
            phi_cmap=viz.DEFAULT_PHI_CMAP,
            u_cmap=viz.DEFAULT_U_CMAP,
            window_size=(800, 600),
            bg_bottom=viz.DEFAULT_BG_BOTTOM,
            bg_top=viz.DEFAULT_BG_TOP,
            silhouette=True,
            skip_saturated=False,
        )
        png_s = out_single / "frame_single.png"
        ok = viz.render_frame(frames[0], png_s, scan, cfg_s, index=0, total=1)
        _assert(ok, "render_frame (single) returned False")
        _check_png(png_s, "single-layout PNG", expected_size=(800, 600))

        # ── 6. Skip-saturated guard ────────────────────────────────────
        sat_frames = viz.discover_frames(sat_path.parent, "output_*.vts")
        sat_scan = viz.compute_global_scan(sat_frames, progress=False)
        cfg_skip = viz.RenderConfig(
            layout="single",
            iso_value=0.0,
            smooth_iters=5,
            pass_band=0.10,
            phi_cmap=viz.DEFAULT_PHI_CMAP,
            u_cmap=viz.DEFAULT_U_CMAP,
            window_size=(640, 480),
            bg_bottom=viz.DEFAULT_BG_BOTTOM,
            bg_top=viz.DEFAULT_BG_TOP,
            silhouette=False,
            skip_saturated=True,
        )
        skip_dst = tmpdir / "viz_skip" / "should_not_exist.png"
        skip_dst.parent.mkdir(parents=True, exist_ok=True)
        ok = viz.render_frame(sat_frames[0], skip_dst, sat_scan, cfg_skip,
                              index=0, total=1)
        _assert(not ok, "skip_saturated did not skip the saturated frame")
        _assert(not skip_dst.exists(),
                "skip_saturated wrote a PNG when it shouldn't have")
        sys.stdout.write("  PASS  --skip-saturated correctly drops saturated frame\n")

        # ── 7. Scan-JSON dump via CLI ──────────────────────────────────
        scan_json = tmpdir / "scan.json"
        proc = subprocess.run([
            sys.executable, str(cli_path),
            "--input-dir", str(small.parent),
            "--output-dir", str(tmpdir / "viz_cli"),
            "--layout", "single",
            "--window-size", "640", "480",
            "--no-silhouette",
            "--scan-json", str(scan_json),
            "--quiet",
        ], stdout=subprocess.PIPE, text=True)
        sys.stdout.write(proc.stdout)
        _assert(proc.returncode == 0, f"CLI run rc={proc.returncode}")
        _assert(scan_json.exists(), "scan.json was not written")
        with scan_json.open() as fh:
            payload = json.load(fh)
        for key in ("phi_clim", "u_clim", "phi_data_range", "u_data_range",
                    "steps", "solid_fraction", "mean_u", "saturated",
                    "grid_dims", "grid_bounds"):
            _assert(key in payload, f"scan.json missing key {key!r}")
        _assert(len(payload["solid_fraction"]) == len(payload["steps"]),
                "scan.json arrays have inconsistent length")
        _assert(np.allclose(payload["phi_data_range"], phi_range, rtol=0, atol=1e-12),
                f"scan.json phi_data_range {payload['phi_data_range']} "
                f"!= synthetic {phi_range}")
        _assert(np.allclose(payload["u_data_range"], u_range, rtol=0, atol=1e-12),
                f"scan.json u_data_range {payload['u_data_range']} "
                f"!= synthetic {u_range}")
        u_line = (f"u   data range [{u_range[0]:.5f}, {u_range[1]:.5f}], "
                  f"colour range [{payload['u_clim'][0]:.5f}, "
                  f"{payload['u_clim'][1]:.5f}]")
        _assert(u_line in proc.stdout,
                f"prescan log does not report {u_line!r}")
        sys.stdout.write("  PASS  CLI --scan-json dump (data ranges match synthetic "
                         "min/max)\n")

        # ── 8. MP4 stitching ───────────────────────────────────────────
        # Synthesize 3 dummy frames so imageio has something to stitch.
        videodir = tmpdir / "viz_video"
        videodir.mkdir(parents=True, exist_ok=True)
        for i, tone in enumerate((50, 130, 220)):
            arr = np.full((128, 256, 3), tone, dtype=np.uint8)
            Image.fromarray(arr).save(videodir / f"frame_{i:06d}.png")
        png_paths = sorted(videodir.glob("frame_*.png"))
        mp4 = videodir / "movie.mp4"
        viz.stitch_video(png_paths, mp4, fps=4)
        _assert(mp4.exists(), "MP4 was not produced")
        _assert(mp4.stat().st_size > 1_000,
                f"MP4 suspiciously small: {mp4.stat().st_size} bytes")
        sys.stdout.write(f"  PASS  MP4 stitched -- {mp4.name} "
                         f"({mp4.stat().st_size:,} bytes)\n")

        # ── 9. Re-run into a reused output dir (no stale frames) ───────
        # Simulation A (steps 0/10/20) and then simulation B (steps 0/10,
        # different data, step 10 wall-touching) are rendered into the SAME
        # output dir, as `make cuda-all` followed by `make cuda-dendrite-demo`
        # does. Afterwards every frame there must come from B: step 0 must be
        # byte-identical to B rendered into an empty dir, step 10 (skipped as
        # saturated) must not keep A's PNG, and the MP4 must hold B's frame
        # only — not A's step 20.
        run_a = tmpdir / "rerun_a"
        run_b = tmpdir / "rerun_b"
        for step, r0 in ((0, 2.0), (10, 2.5), (20, 3.0)):
            _make_tmp_grid(run_a, n=24, dx=0.5, r0=r0, name=f"output_{step}.vts")
        for step, r0 in ((0, 4.0), (10, 12.0)):
            _make_tmp_grid(run_b, n=24, dx=0.5, r0=r0, name=f"output_{step}.vts")

        def _render_cli(in_dir: Path, out_dir: Path) -> None:
            rc = subprocess.call([
                sys.executable, str(cli_path),
                "--input-dir", str(in_dir),
                "--output-dir", str(out_dir),
                "--layout", "single",
                "--window-size", "320", "240",
                "--no-silhouette",
                "--skip-saturated",
                "--make-video", "--fps", "4",
                "--quiet",
            ])
            _assert(rc == 0, f"CLI run {in_dir} -> {out_dir} rc={rc}")

        shared = tmpdir / "viz_shared"
        fresh_b = tmpdir / "viz_fresh_b"
        _render_cli(run_a / "out", shared)
        a_frames = {p.name: p.read_bytes() for p in shared.glob("frame_*.png")}
        _assert(sorted(a_frames) == ["frame_000000.png", "frame_000010.png",
                                     "frame_000020.png"],
                f"simulation A rendered unexpected frames: {sorted(a_frames)}")
        _render_cli(run_b / "out", shared)
        _render_cli(run_b / "out", fresh_b)

        got = (shared / "frame_000000.png").read_bytes()
        _assert(got != a_frames["frame_000000.png"],
                "frame_000000.png in reused output dir is still simulation A's "
                "frame (stale frame reused)")
        _assert(got == (fresh_b / "frame_000000.png").read_bytes(),
                "frame_000000.png in reused output dir differs from a fresh "
                "render of simulation B")
        _assert(not (shared / "frame_000010.png").exists(),
                "frame_000010.png from simulation A survived although B's "
                "step 10 was skipped as saturated")
        n_video = sum(1 for _ in imageio.get_reader(str(shared / "dendrite.mp4")))
        _assert(n_video == 1,
                f"MP4 in reused output dir has {n_video} frames, expected 1 "
                "(simulation B's step 0 only)")
        sys.stdout.write("  PASS  re-run into reused output dir renders "
                         "current input only (no stale frames)\n")

        sys.stdout.write("==> visualize_smoke: ALL CHECKS PASSED\n")
        return 0
    finally:
        # Clean up tmp tree unless KEEP_TMP=1
        if os.environ.get("KEEP_TMP") != "1":
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            sys.stdout.write(f"    (kept tmp tree at {tmpdir})\n")


if __name__ == "__main__":
    raise SystemExit(main())
