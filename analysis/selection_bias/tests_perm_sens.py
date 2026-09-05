import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
here = Path(__file__).resolve().parent
b = pd.read_csv(here/"blocks.csv"); d = b[b.disturbed].copy()
e = pd.read_csv(here/"envs.csv"); ed = e[e.disturbed & e.eligible].copy()

print("--- pre-onset env-pair discrepancies: direct recount (an earlier draft said 2; it is not) ---")
print("disturbed env pairs:", len(e[e.disturbed]), " eligible:", len(ed))
allb = b
print("pre-onset fall differing env pairs, ALL blocks   :", int(allb.pre_fall_env_diff.sum()))
print("pre-onset fall differing env pairs, DISTURBED    :", int(d.pre_fall_env_diff.sum()))
print("blocks affected (disturbed):", int((d.pre_fall_env_diff>0).sum()),
      " of which leak-free:", int(((d.pre_fall_env_diff>0)&d.leakfree).sum()))
print("onset_obs maxdiff among leak-free blocks (must be all 0):",
      float(d[d.leakfree].onset_obs_maxdiff.max()))

print("\n--- PERMUTATION: randomize arm label within env pair, leak-free subset only ---")
rng = np.random.default_rng(11)
for cell, g in ed.groupby("cell"):
    lf = g[g.leakfree]
    u = lf.u_post.values.astype(np.int8); o = lf.o_post.values.astype(np.int8)
    blk = (lf.replicate + "|" + lf.block_id.astype(str)).values
    rep = lf.replicate.values
    df = pd.DataFrame(dict(rep=rep, blk=blk, u=u, o=o))
    def stat(uu, oo):
        t = pd.DataFrame(dict(rep=rep, blk=blk, u=uu, o=oo))
        bm = t.groupby(["rep","blk"]).apply(lambda x: x.u.mean()-x.o.mean(), include_groups=False)
        return bm.groupby(level=0).mean().mean()
    obs = stat(u, o)
    null = np.empty(5000)
    for i in range(5000):
        s = rng.random(len(u)) < 0.5
        null[i] = stat(np.where(s,o,u), np.where(s,u,o))
    p = (np.abs(null) >= abs(obs)).mean()
    print(" %-12s observed=%+6.2f pp  perm-null mean=%+.3f sd=%.3f  two-sided p=%.4g (5000 perms, min resolvable 2e-4)"
          % (cell, obs*100, null.mean()*100, null.std()*100, max(p, 1/5000)))

print("\n--- SENSITIVITY: adversarial values for the CONTAMINATED (excluded) blocks ---")
print("bound the FULL-sample per-cell effect if every contaminated block took its most adverse value")
for cell, g in d.groupby("cell"):
    lf = g[g.leakfree]; dv = g[~g.leakfree]
    # replicate-grouped mean; contaminated block effects forced to -1 / +1 (pp bounds of a rate difference)
    def mixed(fill):
        vals=[]
        for rep, gg in g.groupby("replicate"):
            v = np.where(gg.leakfree.values, gg.effect.values, fill)
            vals.append(np.nanmean(v))
        return np.mean(vals)*100
    print(" %-12s registered=%+6.2f  worst-case-low=%+7.2f  worst-case-high=%+7.2f  (%d/%d blocks excluded)"
          % (cell, g.effect.mean()*100, mixed(-1.0), mixed(1.0), len(dv), len(g)))

print("\nconstrained sensitivity: contamination flips at most k eligible env-pair outcomes per replicate,")
print("calibrated by the observed pre-onset discrepancy budget (k = observed differing env pairs per replicate)")
budget = d.groupby("replicate").pre_fall_env_diff.sum()
print("observed per-replicate pre-onset discrepancy budget:", budget.to_dict())
for cell, g in d.groupby("cell"):
    deltas=[]
    for rep, gg in g.groupby("replicate"):
        k = int(budget.get(rep,0))
        n_elig = gg.n_eligible.sum()
        # most adverse: flip k outcomes all in the direction that shrinks the effect
        deltas.append(k / max(n_elig,1))
    shift = np.mean(deltas)*100
    print(" %-12s max |shift| from a k-flip budget = %.3f pp  (registered %+.2f pp)" % (cell, shift, g.effect.mean()*100))
