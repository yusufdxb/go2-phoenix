"""Build a block-level frame for the 12 v2 replicates: design vars, subset membership, outcomes.

Read-only on the frozen artifacts. Writes blocks.csv + envs.csv under this directory.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/yusuf/workspace/go2-phoenix")
sys.path.insert(0, str(ROOT / "src"))
from phoenix.reliability.replication import _read_arm, _ordered_arm, read_registry  # noqa
from phoenix.reliability.study import read_protocol  # noqa

# Default is the frozen n=3 registry so the file reproduces the original frame
# unchanged; pass a registry path to build the frame over more replicates.
REG = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else ROOT / "reliability_eval/causal_viability_replication_v2/registry.json"
)
reg = read_registry(REG)
root = REG.parent

brows, erows = [], []
for entry in reg["entries"]:
    out = root / entry["out_dir"]
    blocks, proto = read_protocol(out / "protocol.json")
    p = proto["params"]
    envs = int(p["envs_per_block"]); nb = len(blocks)
    bid = np.asarray([b.block_id for b in blocks], dtype=np.int64)
    arms = {}
    for arm in ("unshielded", "oracle"):
        raw, _ = _read_arm(out, arm)
        arms[arm] = _ordered_arm(raw, arm=arm, block_ids=bid, n_blocks=nb, envs=envs)
    u, o = arms["unshielded"], arms["oracle"]
    # EXACT replication of the selection predicate in onset_residual._divergent_blocks
    diff = np.abs(u["onset_obs"] - o["onset_obs"])
    divergent = diff.reshape(nb, -1).max(axis=1) > 0
    onset = np.asarray([b.onset_tick for b in blocks])
    dist = np.asarray([b.disturbed for b in blocks], dtype=bool)
    joint = dist[:, None] & ~u["pre_onset_fall"] & ~o["pre_onset_fall"]
    nelig = joint.sum(axis=1)
    def rate(v):
        return np.divide((v & joint).sum(1), nelig, out=np.full(nb, np.nan), where=nelig > 0)
    ru, ro = rate(u["post_onset_fall"]), rate(o["post_onset_fall"])
    for i, b in enumerate(blocks):
        brows.append(dict(
            cell=p["cell_id"], replicate=p["replicate_id"], policy=p["policy_name"],
            family=p["disturbance_kind"], block_id=int(b.block_id), block_index=i,
            onset=int(b.onset_tick), disturbed=bool(b.disturbed), seed=int(b.seed),
            obs_noise=b.obs_noise, motor_scale=b.motor_scale, command_speed=b.command_speed,
            divergent=bool(divergent[i]), leakfree=bool(~divergent[i]),
            onset_obs_maxdiff=float(diff.reshape(nb, -1).max(axis=1)[i]),
            n_eligible=int(nelig[i]),
            u_post_rate=float(ru[i]), o_post_rate=float(ro[i]),
            effect=float(ru[i] - ro[i]),
            u_pre_rate=float(u["pre_onset_fall"][i].mean()),
            o_pre_rate=float(o["pre_onset_fall"][i].mean()),
            pre_effect=float(u["pre_onset_fall"][i].mean() - o["pre_onset_fall"][i].mean()),
            u_fell=float(u["fell"][i].mean()), o_fell=float(o["fell"][i].mean()),
            u_falltick_med=float(np.median(u["fall_tick"][i])),
            u_task=float(u["task_complete"][i].mean()), o_task=float(o["task_complete"][i].mean()),
            pre_fall_env_diff=int(np.abs(u["pre_onset_fall"][i].astype(int)
                                         - o["pre_onset_fall"][i].astype(int)).sum()),
        ))
        for j in range(envs):
            erows.append(dict(
                cell=p["cell_id"], replicate=p["replicate_id"], block_id=int(b.block_id),
                env=j, onset=int(b.onset_tick), disturbed=bool(b.disturbed),
                leakfree=bool(~divergent[i]), eligible=bool(joint[i, j]),
                u_post=bool(u["post_onset_fall"][i, j]), o_post=bool(o["post_onset_fall"][i, j]),
                u_pre=bool(u["pre_onset_fall"][i, j]), o_pre=bool(o["pre_onset_fall"][i, j]),
                u_falltick=int(u["fall_tick"][i, j]), o_falltick=int(o["fall_tick"][i, j]),
            ))
here = Path(__file__).parent
pd.DataFrame(brows).to_csv(here / "blocks.csv", index=False)
pd.DataFrame(erows).to_csv(here / "envs.csv", index=False)
print("blocks", len(brows), "envs", len(erows))
