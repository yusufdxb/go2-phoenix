"""ADVERSARIAL ATTACK 2: differential post-onset exposure window.

Leak-free blocks have onset ~121.5, divergent ~164.3 => post-onset windows of
378 vs 336 ticks. The outcome is "fell in (onset, 500]". A longer window
mechanically permits more falls. If the two arms have DIFFERENT post-onset
hazard shapes, a longer window biases the DIFFERENCE, not just the levels.

Fix: recompute the primary estimand on a COMMON-LENGTH window
   post_onset_fall_W = onset <= fall_tick < onset + W
for W = 300 (= 500 - max onset 200), which every block can supply in full.
If the effects are unchanged, exposure length is not doing the work.
"""
import numpy as np, pandas as pd
from pathlib import Path
here = Path(__file__).parent
e = pd.read_csv(here/"envs.csv"); b = pd.read_csv(here/"blocks.csv")
d = e[e.disturbed].copy()
print("onset range over all blocks:", b.onset.min(), b.onset.max(), " horizon = 500")
W_MAX = 500 - b.onset.max()
print("largest window every block can supply in full: W =", W_MAX)

def effects_at_W(W):
    x = d.copy()
    up = (x.u_falltick >= x.onset) & (x.u_falltick < x.onset + W) & (x.u_falltick >= 0)
    op = (x.o_falltick >= x.onset) & (x.o_falltick < x.onset + W) & (x.o_falltick >= 0)
    x["uW"], x["oW"] = up, op
    x = x[x.eligible]
    g = x.groupby(["cell","replicate","block_id","leakfree"]).agg(uW=("uW","mean"), oW=("oW","mean")).reset_index()
    g["eff"] = g.uW - g.oW
    return g

print("\n%-12s %10s %10s %10s %10s" % ("cell","W","FULL pp","LEAKFREE","DIVERGENT"))
for W in (300, 336, 378, 400, 500):
    g = effects_at_W(W)
    for cell, gg in g.groupby("cell"):
        mm = lambda s: np.mean([v.eff.mean() for _, v in s.groupby("replicate")])*100
        print("%-12s %10d %10.2f %10.2f %10.2f" % (cell, W, mm(gg), mm(gg[gg.leakfree]), mm(gg[~gg.leakfree])))
    print()

print("=== Is the exposure effect DIFFERENTIAL across arms? ===")
print("Per-cell post-onset hazard by elapsed-since-onset decile, both arms, on")
print("ELIGIBLE disturbed env pairs. If the arms' hazards have the same SHAPE,")
print("a longer window changes both levels but not the difference sign/size.")
x = d[d.eligible].copy()
for cell, g in x.groupby("cell"):
    for arm, ft in (("unsh","u_falltick"), ("orac","o_falltick")):
        el = g[ft] - g.onset
        hit = (g[ft] >= 0) & (el >= 0)
        cum = [float(((el <= t) & hit).mean()) for t in (50,100,150,200,250,300)]
        print("  %-11s %s cumulative post-onset fall frac @t=50..300: %s" %
              (cell, arm, " ".join("%.3f" % c for c in cum)))
    # marginal contribution of the tail beyond W=300
    tailu = float((((g.u_falltick - g.onset) >= 300) & (g.u_falltick >= 0)).mean())
    tailo = float((((g.o_falltick - g.onset) >= 300) & (g.o_falltick >= 0)).mean())
    print("  %-11s falls occurring at elapsed>=300: unsh=%.4f orac=%.4f  diff=%+.4f pp" %
          (cell, tailu, tailo, (tailu-tailo)*100))

print("\n=== Direct test: within the LEAK-FREE subset only, truncate to the")
print("=== window length the DIVERGENT blocks actually get (336 ticks).")
print("=== If the subset effect moves toward the complement, exposure explains the gap.")
g300 = effects_at_W(300); gfull = effects_at_W(500); g336 = effects_at_W(336)
for cell in sorted(b.cell.unique()):
    mm = lambda s: np.mean([v.eff.mean() for _, v in s.groupby("replicate")])*100
    lf5, lf3, lf336 = [mm(t[(t.cell==cell)&t.leakfree]) for t in (gfull,g300,g336)]
    dv5, dv3 = [mm(t[(t.cell==cell)&~t.leakfree]) for t in (gfull,g300)]
    print(" %-12s LF: full=%+6.2f W336=%+6.2f W300=%+6.2f | DIV: full=%+6.2f W300=%+6.2f | gap full=%+6.2f gap W300=%+6.2f"
          % (cell, lf5, lf336, lf3, dv5, dv3, lf5-dv5, lf3-dv3))
