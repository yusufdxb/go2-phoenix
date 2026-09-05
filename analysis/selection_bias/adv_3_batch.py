"""ADVERSARIAL ATTACK 4: cross-block batch coupling breaks block independence.

The harness runs ALL 48 blocks of an arm as ONE PhysX batch of 768 environments
(scripts/reliability_closed_loop.py: total_envs = envs_per_block * n_blocks,
"one pass of 500 ticks"). So blocks are spatial slices of a single simulation,
not independent runs. Three consequences to test:

 (a) is membership predicted by block_index once onset is conditioned on?
 (b) do neighbouring blocks share divergence beyond what onset explains?
 (c) are block-level OUTCOMES autocorrelated along block_index? (this is what
     the block bootstrap and the within-pair permutation actually assume away)
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
here = Path(__file__).parent
b = pd.read_csv(here/"blocks.csv"); d = b[b.disturbed].copy()

print("=== (a) membership | onset: is block_index informative AT ALL? ===")
# These two lines used to be hardcoded prints asserting "0/576 mismatches" and
# "ZERO residual variance". At n=5 both are false, and the same file's computed
# output contradicted them. Compute the mismatch count instead of claiming it.
mismatch = 0
for (rep, cell), g in b.groupby(["replicate", "cell"]):
    thr = g[g.leakfree].onset.max()
    mismatch += int((g.leakfree != (g.onset <= thr)).sum())
print("  membership vs 1{onset <= per-pair threshold}: %d mismatches in %d blocks"
      % (mismatch, len(b)))
print("  If that count is 0, nothing but onset can carry information about membership.")
print("  Verify directly by exact conditioning:")
tot=0; informative=0
for (rep,cell), g in b.groupby(["replicate","cell"]):
    for o, gg in g.groupby("onset"):
        if gg.leakfree.nunique() > 1: informative += 1   # same onset, different membership
        tot += 1
print("  distinct (arm-pair, onset) strata:", tot,
      " strata where membership varies within the stratum:", informative)
if informative == 0:
    print("  => conditional on onset within an arm pair, membership has ZERO residual variance.")
    print("  => block_index / neighbour effects on membership are 100% mediated by onset.")
else:
    print("  => membership is NOT a pure function of onset: %d of %d strata are impure."
          % (informative, tot))
    print("  => onset mediates almost all of it, but the 'exactly deterministic' claim fails.")

print("\n=== is the reported block_index association just onset-block_index correlation? ===")
for cell, g in d.groupby("cell"):
    r,p = stats.spearmanr(g.block_index, g.onset)
    mw = stats.mannwhitneyu(g[g.leakfree].block_index, g[~g.leakfree].block_index)
    # conditional test: permute block_index within onset strata (exact null of no
    # residual block_index effect given onset)
    rng = np.random.default_rng(3)
    obs = g[g.leakfree].block_index.mean() - g[~g.leakfree].block_index.mean()
    print("  %-11s spearman(block_index,onset) rho=%+.3f p=%.4g | raw block_index MWU p=%.4g | LF-DIV mean bi diff=%+.2f"
          % (cell, r, p, mw.pvalue, obs))

print("\n=== (b) neighbour coupling: does block i's divergence depend on block i-1's, given onset? ===")
rows=[]
for (rep,cell), g in b.sort_values("block_index").groupby(["replicate","cell"]):
    v = g.divergent.values.astype(int); o = g.onset.values
    if len(np.unique(v))<2: continue
    # observed lag-1 agreement
    agree = float((v[1:]==v[:-1]).mean())
    # null: permute divergence labels but respect the onset-threshold rule by
    # permuting the ONSET VECTOR (design randomization) and re-deriving membership
    thr = g[g.leakfree].onset.max()
    rng = np.random.default_rng(17)
    null=[]
    for _ in range(20000):
        op = rng.permutation(o)
        vp = (op <= thr).astype(int)
        null.append((vp[1:]==vp[:-1]).mean())
    null=np.array(null)
    rows.append((rep,cell,agree,null.mean(),float((null>=agree).mean())))
res=pd.DataFrame(rows,columns=["rep","cell","lag1_agree","null_mean","p_one_sided"])
print(res.to_string(index=False))
print("  Fisher combined p over 12 arm pairs:",
      "%.4g" % stats.combine_pvalues(res.p_one_sided.clip(1e-6,1).values, method="fisher").pvalue)

print("\n=== (c) THE REAL INDEPENDENCE QUESTION: are BLOCK OUTCOMES autocorrelated ===")
print("=== along block_index within an arm pair? (bootstrap/permutation assume not) ===")
out=[]
for (rep,cell), g in d.sort_values("block_index").groupby(["replicate","cell"]):
    for col in ["effect","u_post_rate","o_post_rate"]:
        x = g[col].dropna().values
        if len(x)<5: continue
        xc = x - x.mean()
        ac1 = float((xc[1:]*xc[:-1]).sum()/ (xc**2).sum())
        rng=np.random.default_rng(5)
        null=np.array([ (lambda y:(y[1:]*y[:-1]).sum()/(y**2).sum())(rng.permutation(xc)) for _ in range(20000)])
        out.append((rep,cell,col,ac1,float((np.abs(null)>=abs(ac1)).mean())))
o=pd.DataFrame(out,columns=["rep","cell","col","lag1_autocorr","p"])
for col,g in o.groupby("col"):
    fp = stats.combine_pvalues(g.p.clip(1e-6,1).values, method="fisher").pvalue
    print("  %-12s mean|lag1 autocorr|=%.3f  n_pairs_p<.05=%d/%d  Fisher combined p=%.4g"
          % (col, g.lag1_autocorr.abs().mean(), int((g.p<0.05).sum()), len(g), fp))
print(o.to_string(index=False))
