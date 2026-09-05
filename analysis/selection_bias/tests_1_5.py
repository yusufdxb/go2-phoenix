import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
here = Path("/home/yusuf/workspace/go2-phoenix/analysis/selection_bias")
b = pd.read_csv(here/"blocks.csv")
d = b[b.disturbed].copy()   # disturbed blocks = the primary estimand's analysis set

print("="*70); print("SHAPE: total blocks", len(b), " disturbed", len(d), " env pairs(dist)", len(d)*16)
print("leak-free (bit-identical) blocks: all=%d/%d  disturbed=%d/%d"
      % (b.leakfree.sum(), len(b), d.leakfree.sum(), len(d)))

print("\n--- TEST 1: membership x arm/cell ---")
ct = pd.crosstab(d.cell, d.leakfree)
print(ct)
chi2,p,dof,_ = stats.chi2_contingency(ct)
print("chi2=%.3f dof=%d p=%.4g" % (chi2,p,dof))
print("rate per cell:\n", d.groupby("cell").leakfree.mean())
ct2 = pd.crosstab(d.replicate, d.leakfree); chi2b,pb,_,_ = stats.chi2_contingency(ct2)
print("by replicate chi2=%.3f p=%.4g" % (chi2b,pb)); print(d.groupby("replicate").leakfree.mean())
print("NOTE: membership is defined jointly on the PAIR, so it is identical for both arms")
print("within a pair by construction -> no unshielded-vs-oracle imbalance is possible.")
print("per-arm-pair (replicate x cell) leak-free block counts (disturbed):")
print(d.groupby(["replicate","cell"]).leakfree.agg(['sum','count']).T)
print("all-blocks leak-free counts per arm pair (the '4-42 of 48' claim, DIVERGENT count):")
print(b.groupby(["replicate","cell"]).divergent.sum().describe())

print("\n--- TEST 2: membership x onset tick ---")
for grp,g in d.groupby(["replicate","cell"]):
    lf, dv = g[g.leakfree].onset, g[~g.leakfree].onset
    sep = (lf.max() < dv.min()) if len(lf) and len(dv) else None
    print(" %-11s %-11s nLF=%2d onsetLF[%3d,%3d] onsetDIV[%3d,%3d] sep=%s"
          % (grp[0],grp[1],len(lf),lf.min(),lf.max(),dv.min(),dv.max(),sep))
auc = stats.mannwhitneyu(d[d.leakfree].onset, d[~d.leakfree].onset, alternative="less")
print("pooled Mann-Whitney U p=%.3g ; AUC(onset predicts divergence)=%.4f"
      % (auc.pvalue, auc.statistic/(d.leakfree.sum()*(~d.leakfree).sum())))
print("mean onset leak-free=%.1f  divergent=%.1f" % (d[d.leakfree].onset.mean(), d[~d.leakfree].onset.mean()))

print("\n--- TEST 3: membership x block index / seed / dose ---")
print("block_index: pointbiserial r=%.4f p=%.3g" % stats.pointbiserialr(d.leakfree, d.block_index))
print("seed:        pointbiserial r=%.4f p=%.3g" % stats.pointbiserialr(d.leakfree, d.seed))
for cov in ["obs_noise","motor_scale","command_speed"]:
    s = d.dropna(subset=[cov])
    if len(s)==0: continue
    r,p = stats.pointbiserialr(s.leakfree, s[cov])
    t = stats.mannwhitneyu(s[s.leakfree][cov], s[~s.leakfree][cov])
    print(" %-13s n=%3d r=%+.4f p=%.3g  MWU p=%.3g  meanLF=%.4f meanDIV=%.4f"
          % (cov,len(s),r,p,t.pvalue,s[s.leakfree][cov].mean(),s[~s.leakfree][cov].mean()))

print("\n--- TEST 4: membership x OUTCOME and pre-outcome covariates ---")
print("[control-arm potential outcome Y(0) = unshielded post-onset fall rate]")
for col,label in [("u_post_rate","Y(0) unshielded post-onset fall rate"),
                  ("o_post_rate","Y(1) oracle post-onset fall rate"),
                  ("u_pre_rate","PRE-treatment: unshielded pre-onset fall rate"),
                  ("o_pre_rate","PRE-treatment: oracle pre-onset fall rate"),
                  ("n_eligible","PRE-treatment: eligible env pairs per block"),
                  ("effect","EFFECT (u-o)")]:
    s = d.dropna(subset=[col])
    lf, dv = s[s.leakfree][col], s[~s.leakfree][col]
    r,p = stats.pointbiserialr(s.leakfree, s[col])
    mwu = stats.mannwhitneyu(lf,dv)
    print(" %-52s LF=%.4f DIV=%.4f  r=%+.4f p=%.3g MWU p=%.3g" % (label,lf.mean(),dv.mean(),r,p,mwu.pvalue))
print("\nwithin-cell (stratified) outcome association:")
for cell,g in d.groupby("cell"):
    for col in ["u_post_rate","o_post_rate","effect","u_pre_rate"]:
        s=g.dropna(subset=[col]); r,p = stats.pointbiserialr(s.leakfree,s[col])
        print("  %-11s %-12s LF=%.4f DIV=%.4f r=%+.4f p=%.3g"
              % (cell,col,s[s.leakfree][col].mean(),s[~s.leakfree][col].mean(),r,p))

print("\n--- effect-modification by onset tick (the mechanism that would make the subset non-representative) ---")
for cell,g in d.groupby("cell"):
    s=g.dropna(subset=["effect"])
    r,p = stats.pearsonr(s.onset, s.effect); rs,ps = stats.spearmanr(s.onset, s.effect)
    sl,ic,_,pv,_ = stats.linregress(s.onset, s.effect)
    print("  %-11s pearson r=%+.3f p=%.3g spearman=%+.3f p=%.3g slope=%+.5f pp/tick p=%.3g"
          % (cell,r,p,rs,ps,sl*100,pv))
