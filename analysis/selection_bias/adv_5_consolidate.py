"""Consolidation: the two objections that survived, with CIs.

(A) Administrative-censoring sensitivity. The outcome is 'fell before tick 500',
    so the post-onset window is 500 - onset and varies 300..400 ticks BY DESIGN.
    Recompute the primary estimand on a common 300-tick window (the longest every
    block can supply) and put CIs on it, full and leak-free.
(B) Process-level inference. Frozen CIs never resample the 3 processes.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
here=Path(__file__).parent
e=pd.read_csv(here/"envs.csv"); b=pd.read_csv(here/"blocks.csv")
d=e[e.disturbed & e.eligible].copy()

def blockframe(W):
    x=d.copy()
    x["uW"]=(x.u_falltick>=x.onset)&(x.u_falltick<x.onset+W)&(x.u_falltick>=0)
    x["oW"]=(x.o_falltick>=x.onset)&(x.o_falltick<x.onset+W)&(x.o_falltick>=0)
    g=x.groupby(["cell","replicate","block_id","leakfree"],dropna=False) \
        .agg(u=("uW","mean"),o=("oW","mean")).reset_index()
    g["eff"]=g.u-g.o
    return g

def boot(groups,n_boot=20000,seed=20260830):
    rng=np.random.default_rng(seed); groups=[np.asarray(g,float) for g in groups if len(g)]
    pt=float(np.mean(np.concatenate(groups)))
    dr=np.array([np.mean(np.concatenate([g[rng.integers(0,len(g),len(g))] for g in groups])) for _ in range(n_boot)])
    return pt,float(np.percentile(dr,2.5)),float(np.percentile(dr,97.5))

print("=== (A) primary estimand under a COMMON 300-tick post-onset window ===")
print("%-12s %-10s %-26s %-26s %s" % ("cell","subset","W=500 (registered)","W=300 (common window)","shift"))
g5=blockframe(500); g3=blockframe(300)
for cell in sorted(b.cell.unique()):
    for name,sel in [("FULL",lambda t:t),("LEAKFREE",lambda t:t[t.leakfree])]:
        a=sel(g5[g5.cell==cell]); c=sel(g3[g3.cell==cell])
        p5,l5,h5=boot([v.eff.values for _,v in a.groupby("replicate")])
        p3,l3,h3=boot([v.eff.values for _,v in c.groupby("replicate")])
        print("%-12s %-10s %+7.2f [%+7.2f,%+7.2f]  %+7.2f [%+7.2f,%+7.2f]  %+6.2f pp %s"
              %(cell,name,p5*100,l5*100,h5*100,p3*100,l3*100,h3*100,(p3-p5)*100,
                "SIGN HOLDS" if np.sign(p3)==np.sign(p5) and l3*h3>0 else "*** BREAKS ***"))

print("\n=== (B) subset-minus-complement gap under the common window, with CI ===")
for cell in sorted(b.cell.unique()):
    for lbl,gg in [("W=500",g5),("W=300",g3)]:
        t=gg[gg.cell==cell]
        A=[v.eff.values for _,v in t[t.leakfree].groupby("replicate")]
        B=[v.eff.values for _,v in t[~t.leakfree].groupby("replicate")]
        rng=np.random.default_rng(7)
        pt=np.mean(np.concatenate(A))-np.mean(np.concatenate(B))
        dr=np.array([np.mean(np.concatenate([x[rng.integers(0,len(x),len(x))] for x in A]))
                     -np.mean(np.concatenate([x[rng.integers(0,len(x),len(x))] for x in B])) for _ in range(20000)])
        lo,hi=np.percentile(dr,[2.5,97.5])
        print("  %-12s %s  delta=%+6.2f pp [%+6.2f,%+6.2f]  crosses zero=%s"
              %(cell,lbl,pt*100,lo*100,hi*100,bool(lo*hi<0)))

print("\n=== (B2) fault-family gate criterion under the common window ===")
fam=dict(stand_motor="motor",walk_motor="motor",stand_obs="obs",walk_obs="obs")
for W,gg in [(500,g5),(300,g3)]:
    t=gg.copy(); t["fam"]=t.cell.map(fam)
    for f,h in t.groupby("fam"):
        for name,sel in [("FULL",lambda x:x),("LEAKFREE",lambda x:x[x.leakfree])]:
            s=sel(h); pt,lo,hi=boot([v.eff.values for _,v in s.groupby(["replicate","cell"])])
            ok=(hi<0) if f=="motor" else (lo>0)
            print("  W=%3d family=%-6s %-9s %+7.2f [%+7.2f,%+7.2f]  gate=%s"%(W,f,name,pt*100,lo*100,hi*100,ok))
