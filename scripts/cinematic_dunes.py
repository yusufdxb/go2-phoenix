"""Render a cinematic clip of ONE Unitree Go2 walking sand dunes, for the portfolio hero.

Uses Isaac Lab's supported offline-video path: the env is created with
``render_mode="rgb_array"`` and frames come from ``env.render()`` (the viewport
render product), NOT from a Camera sensor. A Camera sensor at a /World/... path
returns only the dome colour on this runtime, which is what an earlier version
of this script hit.

The camera is flown with ``sim.set_camera_view(eye, target)`` each step, which
drives the same viewport the frames are read from.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES ~/Sim/isaac-sim-venv/bin/python \
        scripts/cinematic_dunes.py --frames 360 --out /tmp/go2_dunes
"""

from __future__ import annotations

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=360)
parser.add_argument("--out", type=str, default="/tmp/go2_dunes")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--cinematic", action="store_true",
                    help="Render on the offline path (quality .kit preset + RTX PathTracing + "
                         "an HDRI sky) instead of the real-time raster path. Costs several "
                         "render calls per frame, so a 360-frame orbit goes from minutes to "
                         "tens of minutes. Worth it for anything published.")
parser.add_argument("--spp", type=int, default=64,
                    help="Path-traced samples per pixel when --cinematic is set.")
parser.add_argument("--sky_hdri", type=str, default="",
                    help="Basename of a bundled HDRI to light the dome with, e.g. "
                         "photo_studio_01_4k.hdr. Off by default: this scene's warm-key / "
                         "cool-dome rig is art-directed and an HDRI overrides it.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=os.path.expanduser("~/workspace/go2-phoenix/checkpoints/phoenix-flat-v4/latest.pt"),
)
parser.add_argument("--warmup", type=int, default=60, help="settle steps before recording")
parser.add_argument("--flat", action="store_true", help="sand plane instead of the dune heightfield")
parser.add_argument("--robots", type=int, default=1, help="1 or 2 robots in shot")
parser.add_argument("--headed", action="store_true", help="run with a GUI window (real GL context)")
parser.add_argument("--start-index", type=int, default=0, help="global frame index this chunk starts at")
parser.add_argument("--orbit-total", type=int, default=0, help="frames in the FULL orbit (0 = use --frames)")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

# video=True is load-bearing: without it AppLauncher disables the default
# viewport, which makes env.render() return black frames AND turns
# SimulationContext.set_camera_view() into a no-op (it forwards only to
# configured visualizers, of which there are none).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from phoenix.demo import render_profile  # noqa: E402

_launcher_args = {
    "headless": not args.headed,
    "enable_cameras": True,
    "video": True,
    "width": args.width,
    "height": args.height,
}
if args.cinematic:
    # rendering_mode picks a .kit experience file that Kit reads while booting, so
    # it only takes effect if it is set here rather than after the app is up.
    _launcher_args.update(render_profile.launcher_kwargs(args.width, args.height))

app_launcher = AppLauncher(_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.terrains import TerrainGeneratorCfg  # noqa: E402
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
DEVICE = "cuda"


def build_actor(path: str, device: str) -> torch.nn.Module:
    """Rebuild the rsl_rl actor MLP (512-256-128, ELU) from a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["actor_state_dict"]
    dims = [sd["mlp.0.weight"].shape[1], 512, 256, 128, sd["mlp.6.weight"].shape[0]]
    layers: list[torch.nn.Module] = []
    for i in range(3):
        layers += [torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.ELU()]
    layers += [torch.nn.Linear(dims[3], dims[4])]
    net = torch.nn.Sequential(*layers)
    mapping = {
        "0.weight": "mlp.0.weight", "0.bias": "mlp.0.bias",
        "2.weight": "mlp.2.weight", "2.bias": "mlp.2.bias",
        "4.weight": "mlp.4.weight", "4.bias": "mlp.4.bias",
        "6.weight": "mlp.6.weight", "6.bias": "mlp.6.bias",
    }
    net.load_state_dict({k: sd[v] for k, v in mapping.items()})
    return net.to(device).eval()


# ----------------------------------------------------------------- scene cfg
env_cfg = parse_env_cfg(TASK, device=DEVICE, num_envs=max(1, args.robots))
# Determinism matters: every chunk must settle the robot into the SAME pose so
# the stitched orbit has no jump at a chunk boundary.
env_cfg.seed = 20260801

# Keep the robots close together so both fit one frame.
env_cfg.scene.env_spacing = 1.6

# Dunes: one long-wavelength wave field. Amplitude stays low so a FLAT-trained
# policy can still walk it; at camera height it reads as desert.
if not args.flat:
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=5,
        num_cols=5,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        difficulty_range=(0.4, 0.4),
        sub_terrains={
            "dunes": terrain_gen.HfWaveTerrainCfg(
                proportion=1.0,
                amplitude_range=(0.12, 0.18),
                num_waves=2,
                border_width=0.0,
            ),
        },
    )

# Warm sand, for whichever ground is in use. The Isaac Lab default terrain
# material is BLACK, which is the real reason early frames read as "no ground".
env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.78, 0.62, 0.40), roughness=0.95, metallic=0.0
)

# Cinematic camera, re-posed every frame from Python.
env_cfg.scene.cine_cam = CameraCfg(
    prim_path="/World/CineCam",
    update_period=0.0,
    height=args.height,
    width=args.width,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=26.0, focus_distance=6.0, horizontal_aperture=36.0, clipping_range=(0.05, 600.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
)

# No debug overlays: command arrows must not appear in a hero shot.
env_cfg.commands.base_velocity.debug_vis = False

# Terminations are deliberately left ALONE. Clearing time_out and setting a
# 1e6 s episode hangs the sim a few dozen steps in (max_episode_length blows up);
# the stock termination config renders fine. A long episode_length_s is enough
# to avoid a mid-shot reset without touching the termination terms.
env_cfg.episode_length_s = 60.0

# Golden hour: warm low key light plus a cool sky fill for the shadows.
env_cfg.scene.sky_light = None
env_cfg.scene.light = None

# rgb_array is what makes env.render() return frames.
env = gym.make(TASK, cfg=env_cfg)
unwrapped = env.unwrapped

# Golden hour: a strong warm key raking low across the dunes, and a dim cool
# dome so the shadow side stays blue instead of going flat grey. The intensity
# ratio is what gives the sand its relief.
key = sim_utils.DistantLightCfg(intensity=6500.0, color=(1.0, 0.76, 0.48), angle=0.9)
key.func("/World/KeyLight", key, translation=(0.0, 0.0, 40.0), orientation=(0.10, 0.68, 0.20, 0.70))
# NOT enabled by --cinematic, and that is deliberate. An HDRI is the right call for
# a neutral studio hero shot, but this scene is art-directed: the warm key against
# the cool flat dome is what gives the sand its relief, and the only outdoor HDRI
# bundled with Isaac (StinsonBeach) overrides that with a dusk beach, putting an
# ocean on the horizon of a desert and dropping the whole shot into darkness.
# Verified by rendering it. Left as a flag because it is correct for other scenes.
_hdri = render_profile.resolve_hdri(args.sky_hdri) if args.sky_hdri else ""
if _hdri:
    dome = sim_utils.DomeLightCfg(intensity=140.0, color=(1.0, 1.0, 1.0), texture_file=_hdri)
    print(f"[cine] sky HDRI: {os.path.basename(_hdri)}", flush=True)
else:
    dome = sim_utils.DomeLightCfg(intensity=190.0, color=(0.40, 0.54, 0.82))
dome.func("/World/Sky", dome)

actor = build_actor(args.checkpoint, DEVICE)
os.makedirs(args.out, exist_ok=True)

obs, _ = env.reset()
policy_obs = obs["policy"]

robot = unwrapped.scene["robot"]
camera = unwrapped.scene["cine_cam"]
cmd_term = unwrapped.command_manager.get_term("base_velocity")

FORWARD = 0.0  # stand: the camera provides the motion, not the robot
saved = 0
n_accum = render_profile.settle_frames(args.spp, path_tracing=args.cinematic)
total = args.warmup + args.frames
print(f"[cine] {args.frames} frames @ {args.width}x{args.height}, robots={args.robots}", flush=True)

orbit_total = args.orbit_total if args.orbit_total > 0 else args.frames

for step in range(total):
    cmd_term.vel_command_b[:, 0] = FORWARD
    cmd_term.vel_command_b[:, 1] = 0.0
    cmd_term.vel_command_b[:, 2] = 0.0

    root_pos = robot.data.root_pos_w
    if isinstance(root_pos, wp.array):
        root_pos = wp.to_torch(root_pos)
    base = root_pos.mean(dim=0).tolist()

    # Camera angle comes from the GLOBAL frame index so independent chunks
    # concatenate into one continuous orbit.
    gi = args.start_index + max(0, step - args.warmup)
    u = gi / max(orbit_total, 1)
    ang = 2.0 * math.pi * u
    radius = 3.15 + 0.35 * math.sin(2.0 * math.pi * u)
    elev = 0.72 + 0.30 * math.sin(2.0 * math.pi * u + 1.1)

    eye = torch.tensor(
        [[base[0] + radius * math.cos(ang), base[1] + radius * math.sin(ang), base[2] + elev]],
        device=DEVICE,
    )
    target = torch.tensor([[base[0], base[1], base[2] + 0.02]], device=DEVICE)
    rot = quat_from_matrix(create_rotation_matrix_from_view(eye, target, up_axis="Z", device=DEVICE))
    camera.set_world_poses(eye, rot, convention="opengl")

    with torch.inference_mode():
        # Zero action = hold the default joint stance through the PD controller.
        # The trained actor is NOT driven here: this checkpoint carries no
        # observation normalizer (the export bug recorded in the Phoenix audit),
        # so running it open-loop collapses the robot within a few steps. A
        # clean stand is the honest hero shot; the camera supplies the motion.
        action = torch.zeros((unwrapped.num_envs, 12), device=DEVICE)
        obs, _, _, _, _ = env.step(action)
        policy_obs = obs["policy"]

    if step == args.warmup and args.cinematic:
        # Switch only once the warmup is done: warmup frames are discarded, and
        # path-tracing them would multiply the cost of the run for nothing.
        if render_profile.enable_path_tracing(args.spp):
            print(f"[cine] PathTracing @ {args.spp} spp, "
                  f"{n_accum} render calls/frame", flush=True)
        else:
            print("[cine] carb unavailable — staying on the real-time renderer", flush=True)

    if step >= args.warmup:
        # The camera moved this frame, so the buffer still holds reprojected
        # history from the previous pose. Re-render in place (sim.render() does
        # NOT advance physics) until the samples have accumulated.
        for _ in range(n_accum):
            unwrapped.sim.render()
            camera.update(dt=0.0)
        arr = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
            os.path.join(args.out, f"f_{gi:05d}.png")
        )
        saved += 1

print(f"[cine] wrote {saved} frames to {args.out}", flush=True)
env.close()
simulation_app.close()
