import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
here = Path("/home/yusuf/workspace/go2-phoenix/analysis/selection_bias")
d = pd.read_csv(here/"blocks.csv"); d = d[d.disturbed]
print("Selection is deterministic in onset. The ONLY route to bias is effect modification")
print("by onset tick. Bound it: bias = slope(effect~onset) * (mean onset full - mean onset subset).")
for cell,g in d.groupby("cell"):
    s=g.dropna(subset=["effect"])
    res = stats.linregress(s.onset, s.effect)
    n=len(s); tcrit = stats.t.ppf(0.975, n-2)
    lo,hi = res.slope-tcrit*res.stderr, res.slope+tcrit*res.stderr
    shift = s.onset.mean() - s[s.leakfree].onset.mean()
    print(" %-12s slope=%+.4f pp/tick [%+.4f,%+.4f]  onset shift=%.1f ticks"
          % (cell,res.slope*100,lo*100,hi*100,shift))
    print("      -> implied extrapolation bias %+.2f pp, 95%% bound [%+.2f,%+.2f] pp (effect is %+.2f pp)"
          % (res.slope*shift*100, min(lo,hi)*shift*100, max(lo,hi)*shift*100, s.effect.mean()*100))
print("\npost-onset exposure window (500 - onset):")
print(d.groupby("leakfree").apply(lambda x:(500-x.onset).mean(), include_groups=False))
