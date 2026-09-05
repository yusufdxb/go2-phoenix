import numpy as np, pandas as pd
from pathlib import Path
here = Path("/home/yusuf/workspace/go2-phoenix/analysis/selection_bias")
b = pd.read_csv(here/"blocks.csv")

print("--- Is membership EXACTLY a function of onset tick within an arm pair? (all 576 blocks) ---")
mis_tot = 0
for (rep,cell), g in b.groupby(["replicate","cell"]):
    lf = g[g.leakfree].onset; dv = g[~g.leakfree].onset
    thr = lf.max() if len(lf) else -1
    pred = g.onset <= thr
    mism = int((pred != g.leakfree).sum()); mis_tot += mism
    ear = g.onset.min()
    print(" %-11s %-11s thr=%3d  earliest_onset=%3d  delay=%2d ticks  nLF=%2d/48  mismatches=%d"
          % (rep,cell,thr,ear,thr-ear,len(lf),mism))
print("TOTAL blocks where membership != 1{onset <= per-pair threshold}:", mis_tot, "/", len(b))

print("\n--- Does the SAME onset threshold hold for NOMINAL (never-treated) blocks? ---")
print("If yes, divergence is batch-level coupling, not the focal block's own treatment.")
for (rep,cell), g in b.groupby(["replicate","cell"]):
    dis = g[g.disturbed]; nom = g[~g.disturbed]
    thr_d = dis[dis.leakfree].onset.max() if dis.leakfree.any() else -1
    ok = int(((nom.onset <= thr_d) == nom.leakfree).sum())
    print("  %-11s %-11s disturbed-thr=%3d  nominal blocks agreeing=%2d/%2d  nominal LF=%2d/%2d"
          % (rep,cell,thr_d,ok,len(nom),nom.leakfree.sum(),len(nom)))

print("\n--- Balance of PRE-TREATMENT design variables inside vs outside the subset, per cell ---")
from scipy import stats
d = b[b.disturbed]
for cell,g in d.groupby("cell"):
    for cov in ["obs_noise","motor_scale","command_speed","block_index","seed"]:
        s = g.dropna(subset=[cov])
        if len(s)==0 or s[cov].nunique()<2: continue
        t = stats.mannwhitneyu(s[s.leakfree][cov], s[~s.leakfree][cov])
        print("  %-11s %-13s LF=%.4f DIV=%.4f  MWU p=%.3f" %
              (cell,cov,s[s.leakfree][cov].mean(),s[~s.leakfree][cov].mean(),t.pvalue))
