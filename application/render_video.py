"""
Convert a PybulletRecorder .pkl simulation file to an MP4 video.

Works fully headless — no GUI required. Uses PyBullet's ER_TINY_RENDERER
(CPU software renderer) by default, or ER_EGL_OPENGL for GPU-accelerated
rendering on Linux with EGL drivers installed.

Usage
-----
# Single file
python application/render_video.py --pkl path/to/simulation_123.pkl

# All .pkl files under a directory
python application/render_video.py --pkl_dir runs/flow_dual/.../simulation_2026-03-26_12-00-00

# Override output path
python application/render_video.py --pkl path/to/simulation_123.pkl --output my_video.mp4

# Higher resolution / framerate
python application/render_video.py --pkl path/to/simulation_123.pkl --width 1920 --height 1080 --fps 60

# GPU-accelerated rendering (requires EGL)
python application/render_video.py --pkl path/to/simulation_123.pkl --egl
"""

import os
import glob
import pickle
import argparse
import concurrent.futures
import functools
import subprocess
import numpy as np
import pkgutil
from tqdm import tqdm

import pybullet as p
import pybullet_data
import cv2


# ---------------------------------------------------------------------------
# Camera defaults — match the demo.py / configure_pybullet settings
# ---------------------------------------------------------------------------
_DEFAULT_YAW = 46.39
_DEFAULT_PITCH = -55.0
_DEFAULT_DIST = 1.9
_DEFAULT_TARGET = [0.0, 0.0, 0.0]


def _build_camera_matrices(width, height, yaw, pitch, dist, target):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=dist,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.1,
        farVal=100.0,
    )
    return view, proj


def render_pkl(
    pkl_path,
    output_path=None,
    width=1280,
    height=720,
    fps=30,
    use_egl=False,
    yaw=_DEFAULT_YAW,
    pitch=_DEFAULT_PITCH,
    dist=_DEFAULT_DIST,
    target=_DEFAULT_TARGET,
):
    """Render one simulation .pkl to an MP4 file."""
    if output_path is None:
        output_path = os.path.splitext(pkl_path)[0] + ".mp4"

    data = pickle.load(open(pkl_path, "rb"))
    if not data:
        tqdm.write(f"[skip] empty recording: {pkl_path}")
        return

    num_frames = len(next(iter(data.values()))["frames"])
    if num_frames == 0:
        tqdm.write(f"[skip] no frames: {pkl_path}")
        return

    # Connect headlessly
    p.connect(p.DIRECT)
    
    renderer_flag = p.ER_TINY_RENDERER
    if use_egl:
        renderer_flag = p.ER_BULLET_HARDWARE_OPENGL
        
        # Load the EGL plugin for headless GPU rendering
        egl = pkgutil.get_loader("eglRenderer")
        if egl is not None:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            print("Warning: eglRenderer plugin not found. Falling back to software rendering.")
            renderer_flag = p.ER_TINY_RENDERER

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load one MultiBody per link visual stored in the recording
    body_map = {}
    for name, obj in data.items():
        color = obj.get("color")
        if color is None:
            rgba = [0.5, 0.5, 0.5, 1.0]
        elif len(color) == 3:
            rgba = list(color) + [1.0]
        else:
            rgba = list(color)

        try:
            vis_id = p.createVisualShape(
                p.GEOM_MESH,
                fileName=obj["mesh_path"],
                meshScale=obj["mesh_scale"],
                rgbaColor=rgba,
            )
            body_id = p.createMultiBody(baseVisualShapeIndex=vis_id)
            body_map[name] = body_id
        except Exception:
            # Skip meshes that fail to load (missing files, etc.)
            body_map[name] = None

    view_matrix, proj_matrix = _build_camera_matrices(
        width, height, yaw, pitch, dist, target
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    fname = os.path.basename(pkl_path)
    try:
        with tqdm(total=num_frames, desc=fname, unit="fr", leave=True, position=None) as pbar:
            for frame_idx in range(num_frames):
                for name, obj in data.items():
                    body_id = body_map.get(name)
                    if body_id is None:
                        continue
                    frame = obj["frames"][frame_idx]
                    p.resetBasePositionAndOrientation(
                        body_id, frame["position"], frame["orientation"]
                    )

                _, _, rgba_px, _, _ = p.getCameraImage(
                    width,
                    height,
                    view_matrix,
                    proj_matrix,
                    renderer=renderer_flag,
                )

                rgb = np.array(rgba_px, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                proc.stdin.write(bgr.tobytes())
                pbar.update(1)

        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            ffmpeg_err = proc.stderr.read().decode(errors="replace")
            tqdm.write(f"  [error] ffmpeg exited {proc.returncode} for {pkl_path}:\n{ffmpeg_err}")
            return
    except BrokenPipeError:
        ffmpeg_err = proc.stderr.read().decode(errors="replace")
        proc.wait()
        tqdm.write(f"  [error] ffmpeg crashed on {pkl_path}:\n{ffmpeg_err}")
        return
    finally:
        try:
            p.disconnect()
        except Exception:
            pass

    tqdm.write(f"  Saved ({num_frames} fr, {num_frames/fps:.1f}s) → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PybulletRecorder .pkl files to MP4 videos (headless)"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pkl", type=str, help="Path to a single .pkl simulation file")
    src.add_argument(
        "--pkl_dir",
        type=str,
        help="Directory to search recursively for .pkl files",
    )

    parser.add_argument("--output", type=str, default=None, help="Output .mp4 path (single-file mode only)")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--egl", action="store_true", help="Use GPU EGL renderer instead of CPU software renderer")
    parser.add_argument("--yaw", type=float, default=_DEFAULT_YAW, help="Camera yaw (degrees)")
    parser.add_argument("--pitch", type=float, default=_DEFAULT_PITCH, help="Camera pitch (degrees)")
    parser.add_argument("--dist", type=float, default=_DEFAULT_DIST, help="Camera distance from target")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel render workers (batch mode only)")

    args = parser.parse_args()

    kwargs = dict(
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_egl=args.egl,
        yaw=args.yaw,
        pitch=args.pitch,
        dist=args.dist,
    )

    if args.pkl:
        render_pkl(args.pkl, output_path=args.output, **kwargs)
    else:
        pkls = sorted(glob.glob(os.path.join(args.pkl_dir, "**", "*.pkl"), recursive=True))
        if not pkls:
            print(f"No .pkl files found under {args.pkl_dir}")
            return
        print(f"Found {len(pkls)} file(s) to render with {args.num_workers} worker(s)")
        _render = functools.partial(render_pkl, **kwargs)
        if args.num_workers <= 1:
            for pkl_path in pkls:
                _render(pkl_path)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                list(ex.map(_render, pkls))


if __name__ == "__main__":
    main()
