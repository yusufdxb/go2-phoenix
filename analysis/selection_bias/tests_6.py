import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
here = Path(__file__).resolve().parent
b = pd.read_csv(here/"blocks.csv"); d = b[b.disturbed].copy()

ct = pd.crosstab(d.cell, d.leakfree)
r = stats.chi2_contingency(ct)
print("TEST1 corrected: chi2=%.3f dof=%d p=%.4g" % (r.statistic, r.dof, r.pvalue))

# Block bootstrap matching phoenix.reliability.replication._bootstrap_mean semantics:
# resample blocks within each replicate group, average group means.
def boot(groups, n_boot=20000, seed=20260830):
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g, float) for g in groups if len(g)]
    if not groups: return (np.nan,)*3
    point = float(np.mean([g.mean() for g in groups]))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = np.mean([g[rng.integers(0, len(g), len(g))].mean() for g in groups])
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

def cell_groups(df, col="effect"):
    return [g[col].dropna().values for _, g in df.groupby("replicate")]

print("\n--- TEST 6: primary effects, pp  (subset vs complement vs full) ---")
print("%-12s | %-26s | %-26s | %-26s" % ("cell","FULL (registered)","LEAK-FREE subset","CONTAMINATED complement"))
res = {}
for cell, g in d.groupby("cell"):
    row = []
    for name, sub in [("full", g), ("lf", g[g.leakfree]), ("dv", g[~g.leakfree])]:
        p, lo, hi = boot(cell_groups(sub))
        n = sub.effect.notna().sum()
        row.append((p*100, lo*100, hi*100, n)); res[(cell,name)] = (p,lo,hi,n)
    print("%-12s | %+6.2f [%+6.2f,%+6.2f] n=%3d | %+6.2f [%+6.2f,%+6.2f] n=%3d | %+6.2f [%+6.2f,%+6.2f] n=%3d"
          % (cell, *row[0], *row[1], *row[2]))

print("\n--- interaction (obs family minus motor family), pp ---")
for name, sel in [("full", lambda x: x), ("lf", lambda x: x[x.leakfree]), ("dv", lambda x: x[~x.leakfree])]:
    sub = sel(d)
    obs = sub[sub.family=="obs"]; mot = sub[sub.family=="motor"]
    # interaction = mean(obs effects) - mean(motor effects), replicate-grouped bootstrap of both
    rng = np.random.default_rng(20260831)
    go = cell_groups(obs); gm = cell_groups(mot)
    pt = np.mean([g.mean() for g in go]) - np.mean([g.mean() for g in gm])
    dr = np.empty(20000)
    for i in range(20000):
        dr[i] = (np.mean([g[rng.integers(0,len(g),len(g))].mean() for g in go])
                 - np.mean([g[rng.integers(0,len(g),len(g))].mean() for g in gm]))
    print(" %-4s %+6.2f [%+6.2f,%+6.2f]  n_obs=%d n_motor=%d"
          % (name, pt*100, np.percentile(dr,2.5)*100, np.percentile(dr,97.5)*100, len(obs), len(mot)))

print("\n--- pre-onset negative control, pp ---")
for cell, g in d.groupby("cell"):
    out=[]
    for name, sub in [("full", g), ("lf", g[g.leakfree]), ("dv", g[~g.leakfree])]:
        p,lo,hi = boot(cell_groups(sub, "pre_effect")); out.append((p*100,lo*100,hi*100))
    print(" %-12s full %+5.2f [%+5.2f,%+5.2f] | lf %+5.2f [%+5.2f,%+5.2f] | dv %+5.2f [%+5.2f,%+5.2f]"
          % (cell,*out[0],*out[1],*out[2]))

print("\n--- difference-in-effect: subset minus complement (paired bootstrap within cell) ---")
for cell, g in d.groupby("cell"):
    rng = np.random.default_rng(7)
    glf = cell_groups(g[g.leakfree]); gdv = cell_groups(g[~g.leakfree])
    pt = np.mean([x.mean() for x in glf]) - np.mean([x.mean() for x in gdv])
    dr = np.array([np.mean([x[rng.integers(0,len(x),len(x))].mean() for x in glf])
                   - np.mean([x[rng.integers(0,len(x),len(x))].mean() for x in gdv]) for _ in range(20000)])
    lo,hi = np.percentile(dr,[2.5,97.5])
    print(" %-12s delta=%+6.2f pp [%+6.2f,%+6.2f]  crosses zero=%s"
          % (cell, pt*100, lo*100, hi*100, bool(lo*hi<0)))
