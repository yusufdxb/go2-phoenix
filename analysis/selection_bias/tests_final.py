import numpy as np, pandas as pd, json
from pathlib import Path
here = Path(__file__).resolve().parent
b = pd.read_csv(here/"blocks.csv"); d = b[b.disturbed].copy()
e = pd.read_csv(here/"envs.csv")

print("--- audit json vs my recomputation (contamination-free cells) ---")
import os
_here = Path(__file__).resolve().parent
_R = Path(__file__).resolve().parents[2] / "reliability_eval/causal_viability_replication_v2"
_n = pd.read_csv(_here/"blocks.csv").replicate.nunique()
# Follow the frame, so an n=5 frame is never audited against the n=3 audit JSON.
AUDIT = _R / os.environ.get(
    "AUDIT", "onset_residual_audit.json" if _n == 3 else f"onset_residual_audit_n{_n}.json")
a = json.load(open(AUDIT))
for cell,v in a["cells"].items():
    print(" %-12s registered %+6.2f [%+6.2f,%+6.2f] n=%3d | clean %+6.2f [%+6.2f,%+6.2f] n=%3d | pre %+5.3f [%+5.3f,%+5.3f]"
      % (cell, v["registered"]["mean_difference"]*100, v["registered"]["ci_low"]*100, v["registered"]["ci_high"]*100,
         v["registered"]["blocks"], v["contamination_free"]["mean_difference"]*100,
         v["contamination_free"]["ci_low"]*100, v["contamination_free"]["ci_high"]*100,
         v["contamination_free"]["blocks"],
         v["pre_onset_negative_control"]["mean_difference"]*100,
         v["pre_onset_negative_control"]["ci_low"]*100, v["pre_onset_negative_control"]["ci_high"]*100))

print("\n--- pre-onset fall-status discrepancies: direct recount from the raw frame ---")
ed = e[e.disturbed]
diff = (ed.u_pre.astype(bool) != ed.o_pre.astype(bool))
print("disturbed env pairs =", len(ed), " differing pre-onset fall status =", int(diff.sum()))
print("by cell:", ed[diff].groupby("cell").size().to_dict())
print("all 9216 env pairs, differing =", int((e.u_pre.astype(bool)!=e.o_pre.astype(bool)).sum()))
print("audit json summary field pre_onset_fall_difference_environments =",
      a["summary"]["pre_onset_fall_difference_environments"])

print("\n--- how much of the estimand the subset discards, in ENV PAIRS ---")
elig = e[e.disturbed & e.eligible]
print("eligible pairs total=%d  in leak-free blocks=%d (%.1f%%)  discarded=%d (%.1f%%)"
      % (len(elig), elig.leakfree.sum(), 100*elig.leakfree.mean(),
         (~elig.leakfree).sum(), 100*(~elig.leakfree).mean()))

print("\n--- leave-one-process-out ON THE LEAK-FREE SUBSET (subset sizes are very unequal) ---")
print("the audit pools with np.concatenate, so the cell with 9/4/4 blocks is process_01-weighted")
for cell,g in d.groupby("cell"):
    lf = g[g.leakfree]
    per = lf.groupby("replicate").effect.agg(['mean','count'])
    print(" %-12s per-process leak-free effect/nblocks: %s   pooled=%+.2f  mean-of-means=%+.2f"
      % (cell, {k:(round(v['mean']*100,2), int(v['count'])) for k,v in per.iterrows()},
         lf.effect.mean()*100, per['mean'].mean()*100))
    for om in per.index:
        k = lf[lf.replicate!=om]
        print("      leave out %s -> %+.2f pp (n=%d)" % (om, k.effect.mean()*100, len(k)))
