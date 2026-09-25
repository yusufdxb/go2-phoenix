#!/usr/bin/env python3
"""Convert a TorchScript actor (e.g. rl_sar's policy.pt) to ONNX and verify parity.

Needs torch + onnx + onnxruntime (CPU is enough). Not imported by anything in
CI: torch is only touched here, inside main().

    python scripts/sim2sim_torchscript_to_onnx.py --pt policy.pt --obs-dim 45 --out policy.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pt", required=True)
    ap.add_argument("--obs-dim", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--rel-tol", type=float, default=1e-6)
    args = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort
    import torch

    mod = torch.jit.load(args.pt, map_location="cpu").eval()
    dummy = torch.zeros(1, args.obs_dim)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        mod,
        (dummy,),
        str(out),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(args.samples, args.obs_dim)).astype(np.float32)
    with torch.no_grad():
        ref = mod(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(["action"], {"obs": x})[0]
    diff = float(np.max(np.abs(ref - got)))
    # float32 accumulation-order noise scales with the output magnitude, so the
    # criterion is relative to the largest output on these inputs.
    rel = diff / max(float(np.max(np.abs(ref))), 1e-12)
    rep = {
        "pt": args.pt,
        "pt_sha256": hashlib.sha256(Path(args.pt).read_bytes()).hexdigest(),
        "onnx": str(out),
        "onnx_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "obs_dim": args.obs_dim,
        "action_dim": int(ref.shape[1]),
        "samples": args.samples,
        "max_abs_diff": diff,
        "max_abs_output": float(np.max(np.abs(ref))),
        "max_abs_diff_over_max_abs_output": rel,
        "rel_tol": args.rel_tol,
        "pass": rel < args.rel_tol,
    }
    print(json.dumps(rep, indent=2))
    out.with_suffix(".parity.json").write_text(json.dumps(rep, indent=2) + "\n")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
