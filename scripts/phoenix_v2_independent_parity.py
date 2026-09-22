#!/usr/bin/env python3
"""Phase D: an INDEPENDENT forward pass of the actor against the shipped ONNX.

``scripts/parity_gate.py`` rebuilds its torch side with the same helpers the export
used, so a bug inside those helpers cancels (its docstring says so). This check shares
nothing with the export path: it reads the raw ``actor_state_dict`` tensors of the
source checkpoint and evaluates ``ELU`` MLP + rsl_rl ``EmpiricalNormalization``
(``(x - mean) / (std + 1e-2)``) in numpy, then compares with ONNX Runtime on the
shipped ``policy.onnx`` over REAL observations: every row of the 2026-09-21/22 GO2
policy parquets, rebuilt through the deploy ``ObservationBuilder``. It also compares the
ONNX graph's normaliser constants with the checkpoint's.

Usage::

    source scripts/_activate.sh
    PYTHONPATH=src python scripts/phoenix_v2_independent_parity.py --out results/phoenix_v2/parity
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EPS = 1e-2  # rsl_rl EmpiricalNormalization default, read from the installed source
CKPT = Path("checkpoints/phoenix-stand-h25-lat-noise/2026-06-22_21-08-20/model_799.pt")
ONNX = Path("checkpoints/phoenix-stand-h25-lat-noise/policy.onnx")
LOGS = Path("/home/yusuf/workspace/go2-phoenix/logs/payload_evidence_20260922")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--deploy-config", type=Path, default=Path("configs/sim2real/deploy_stand_h25_v2.yaml"))
    args = ap.parse_args()
    import onnx
    import onnxruntime as ort
    import torch
    import yaml
    from onnx import numpy_helper

    from phoenix.sim2real.observation import JointOrder, ObservationBuilder
    from phoenix.sim2real.verify_deploy import build_obs_from_parquet

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["actor_state_dict"]
    sd = {k: v.detach().double().numpy() for k, v in sd.items() if hasattr(v, "detach")}
    mean, std = sd["obs_normalizer._mean"][0], sd["obs_normalizer._std"][0]

    def forward(obs):
        x = (obs.astype(np.float64) - mean) / (std + EPS)
        for i in (0, 2, 4):
            x = elu(x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"])
        return x @ sd["mlp.6.weight"].T + sd["mlp.6.bias"]

    dcfg = yaml.safe_load(args.deploy_config.read_text())
    builder = ObservationBuilder(JointOrder(tuple(dcfg["joint_order"])), dcfg["control"]["default_joint_pos"])
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    # Normaliser constants inside the ONNX graph.
    model = onnx.load(str(ONNX))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    consts = {}
    for node in model.graph.node:
        if node.op_type in ("Sub", "Div"):
            for inp in node.input:
                if inp in inits:
                    consts[node.op_type] = inits[inp].reshape(-1).astype(np.float64)
                else:
                    for c in model.graph.node:
                        if c.op_type == "Constant" and inp in c.output:
                            consts[node.op_type] = numpy_helper.to_array(c.attribute[0].t).reshape(-1)
    norm = {
        "onnx_sub_vs_ckpt_mean_max_abs": float(np.abs(consts["Sub"] - mean).max()) if "Sub" in consts else None,
        "onnx_div_vs_ckpt_std_plus_eps_max_abs": float(np.abs(consts["Div"] - (std + EPS)).max()) if "Div" in consts else None,
    }

    batches = []
    worst = 0.0
    for pq in sorted(LOGS.glob("*/*/policy.parquet")):
        obs = np.stack(list(build_obs_from_parquet(pq, builder))).astype(np.float32)
        ref = forward(obs)
        got = sess.run(None, {name: obs})[0].astype(np.float64)
        d = np.abs(ref - got)
        cos = np.sum(ref * got, 1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-12)
        worst = max(worst, float(d.max()))
        batches.append({
            "source": str(pq.relative_to(LOGS)),
            "parquet_sha256": _sha(pq),
            "n": int(obs.shape[0]),
            "max_abs": float(d.max()),
            "cos_worst_sample": float(cos.min()),
            "obs_abs_max": float(np.abs(obs).max()),
        })
    res = {
        "schema": "phoenix-v2-independent-parity/v1",
        "checkpoint": str(CKPT), "checkpoint_sha256": _sha(CKPT),
        "onnx": str(ONNX), "onnx_sha256": _sha(ONNX),
        "onnx_data_sha256": _sha(ONNX.with_suffix(".onnx.data")),
        "normaliser": norm,
        "tol_max_abs": 1e-5,
        "batches": batches,
        "max_abs": worst,
        "passed": bool(worst <= 1e-5 and all(v is not None and v < 1e-6 for v in norm.values())),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "independent_parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("normaliser", "max_abs", "passed")}, indent=1))
    for b in batches:
        print(b["source"], b["n"], f"{b['max_abs']:.2e}", f"{b['cos_worst_sample']:.9f}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
