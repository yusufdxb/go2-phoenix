"""ADVERSARIAL ATTACK 1: is the per-CELL stratification the REGISTERED level?

Read the frozen protocol/registry and the gate code to find the level at which
the primary effect is actually declared, then test the membership-outcome
association AT THAT LEVEL rather than at the convenient one.
"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]
R = ROOT/"reliability_eval/causal_viability_replication_v2"
here = ROOT/"analysis/selection_bias"

p = json.load(open(R/"process_01/stand_motor/protocol.json"))["params"]
print("REGISTERED primary_estimand string from the frozen protocol:")
print("  ", p["primary_estimand"])
print("   eligibility_rule:", p["eligibility_rule"])
print("   analysis_unit  :", p["analysis_unit"])

# Follow the frame: an n=5 frame must be read against the n=5 summary, or this
# script silently compares the registered gate of one sample to the blocks of
# another. SUMMARY overrides; otherwise pick by the replicate count in the frame.
import os
_n = pd.read_csv(here/"blocks.csv").replicate.nunique()
_default = "combined_summary.json" if _n == 3 else f"combined_summary_n{_n}.json"
SUMMARY = R/os.environ.get("SUMMARY", _default)
s = json.load(open(SUMMARY))
print(f"\nGATE CHECKS actually evaluated by analyze_registry ({SUMMARY.name}):")
for k,v in s["gate_checks"].items(): print("   %-45s %s" % (k,v))
print("   gate_passed:", s["gate_passed"])
print("\nAggregation levels present in the frozen summary:")
print("   pooled_cells (policy x family, n=96):     ", list(s["pooled_cells"].keys()))
print("   pooled_fault_families (family, n=192):    ", list(s["pooled_fault_families"].keys()))
for f,v in s["pooled_fault_families"].items():
    print("     %-6s %+7.3f pp [%+7.3f,%+7.3f] blocks=%d" %
          (f, v["mean_difference"]*100, v["ci_low"]*100, v["ci_high"]*100, v["independent_disturbed_blocks"]))
print("\n>>> The CI-based gate criterion is at the FAULT-FAMILY level (motor / obs),")
print(">>> which POOLS stand and walk. 'Per-cell' is one level FINER than the gate.")

b = pd.read_csv(here/"blocks.csv"); d = b[b.disturbed].copy()
print("\n=== membership-outcome association AT THE REGISTERED (fault-family) LEVEL ===")
for col in ["u_post_rate","o_post_rate","effect"]:
    for fam, g in d.groupby("family"):
        s2 = g.dropna(subset=[col])
        r,pv = stats.pointbiserialr(s2.leakfree, s2[col])
        mw = stats.mannwhitneyu(s2[s2.leakfree][col], s2[~s2.leakfree][col])
        print("  family=%-6s %-12s LF=%.4f DIV=%.4f  r=%+.4f p=%.4g  MWU p=%.4g" %
              (fam, col, s2[s2.leakfree][col].mean(), s2[~s2.leakfree][col].mean(), r, pv, mw.pvalue))
print("\n=== and POOLED over everything (the crudest level) ===")
for col in ["u_post_rate","o_post_rate","effect"]:
    s2 = d.dropna(subset=[col]); r,pv = stats.pointbiserialr(s2.leakfree, s2[col])
    print("  pooled %-12s LF=%.4f DIV=%.4f r=%+.4f p=%.4g" %
          (col, s2[s2.leakfree][col].mean(), s2[~s2.leakfree][col].mean(), r, pv))

print("\n=== the gate criterion itself, recomputed on the leak-free subset ===")
def boot(groups, n_boot=20000, seed=20260830):
    rng = np.random.default_rng(seed)
    groups=[np.asarray(g,float) for g in groups if len(g)]
    pt = float(np.mean(np.concatenate(groups)))
    dr = np.array([np.mean(np.concatenate([g[rng.integers(0,len(g),len(g))] for g in groups])) for _ in range(n_boot)])
    return pt, float(np.percentile(dr,2.5)), float(np.percentile(dr,97.5))
for fam, g in d.groupby("family"):
    for name, sub in [("FULL",g),("LEAKFREE",g[g.leakfree]),("DIVERGENT",g[~g.leakfree])]:
        grp=[gg.effect.dropna().values for _,gg in sub.groupby(["replicate","cell"])]
        pt,lo,hi = boot(grp)
        crit = (hi<0) if fam=="motor" else (lo>0)
        print("  family=%-6s %-9s %+7.2f pp [%+7.2f,%+7.2f] n=%3d  gate-criterion-passes=%s" %
              (fam,name,pt*100,lo*100,hi*100,sum(len(x) for x in grp),crit))

print("\n=== direction-in-all-12-process-cells gate check, on the subset ===")
bad=[]
for (rep,cell),g in d[d.leakfree].groupby(["replicate","cell"]):
    fam = g.family.iloc[0]; m = g.effect.mean()
    ok = (m<0) if fam=="motor" else (m>0)
    print("   %-11s %-11s n=%2d effect=%+7.2f pp  direction_ok=%s" % (rep,cell,len(g),m*100,ok))
    if not ok: bad.append((rep,cell))
print("   process-cells violating the registered direction on the subset:", bad)
