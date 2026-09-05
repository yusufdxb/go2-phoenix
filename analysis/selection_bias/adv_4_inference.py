"""ADVERSARIAL ATTACKS 3 + 5 + 6: threshold data-dependence, small-n inference,
process-level variance, and threshold sensitivity."""
import numpy as np, pandas as pd, itertools
from pathlib import Path
from scipy import stats
here = Path(__file__).parent
b = pd.read_csv(here/"blocks.csv"); d = b[b.disturbed].copy()

print("### FIX to adv_3(c): the o_post_rate Fisher p was contaminated by a")
print("### DEGENERATE arm pair (process_01 stand_obs oracle rate is constant ->")
print("### autocorrelation undefined). Recompute excluding degenerate series.")
out=[]
for (rep,cell), g in d.sort_values("block_index").groupby(["replicate","cell"]):
    for col in ["effect","u_post_rate","o_post_rate"]:
        x=g[col].dropna().values
        if len(x)<5 or np.allclose(x.std(),0): continue
        xc=x-x.mean(); ac=float((xc[1:]*xc[:-1]).sum()/(xc**2).sum())
        rng=np.random.default_rng(5)
        null=np.array([(lambda y:(y[1:]*y[:-1]).sum()/(y**2).sum())(rng.permutation(xc)) for _ in range(20000)])
        out.append((rep,cell,col,ac,float((np.abs(null)>=abs(ac)).mean())))
o=pd.DataFrame(out,columns=["rep","cell","col","ac1","p"])
for col,g in o.groupby("col"):
    print("  %-12s n=%2d series  mean|ac1|=%.3f  #p<.05=%d  Fisher p=%.4g"
          % (col,len(g),g.ac1.abs().mean(),int((g.p<0.05).sum()),
             stats.combine_pvalues(g.p.clip(1e-6,1).values,method="fisher").pvalue))

print("\n### ATTACK 3: the threshold is data-dependent. THRESHOLD SENSITIVITY SWEEP.")
print("### If the leak-free result is an artifact of where the threshold happened to")
print("### fall, moving it should move the answer. Recompute per cell on")
print("### subsets {onset <= q} for a grid of q, ignoring the fitted threshold.")
print("%-12s %s" % ("cell", "  ".join("q=%3d" % q for q in range(105,205,10))))
for cell,g in d.groupby("cell"):
    row=[]
    for q in range(105,205,10):
        s=g[g.onset<=q]
        row.append("%+6.2f" % (np.mean([v.effect.mean() for _,v in s.groupby("replicate")])*100) if len(s)>=6 else "     .")
    print("%-12s %s  (full=%+.2f)" % (cell," ".join(row), np.mean([v.effect.mean() for _,v in g.groupby("replicate")])*100))
print("counts per q:")
for cell,g in d.groupby("cell"):
    print("  %-12s %s" % (cell," ".join("%3d" % (g.onset<=q).sum() for q in range(105,205,10))))

print("\n### ATTACK 5: is the permutation test valid at the SMALL n it actually has?")
print("### The registered analysis_unit is 'scenario block'. Permuting ARM LABEL at")
print("### the ENV level (as tests_perm_sens.py does) assumes env-level exchangeability")
print("### INSIDE a block, which the shared batch does not guarantee. Redo it at the")
print("### registered unit: EXACT sign-flip permutation over BLOCK-LEVEL effects.")
for cell,g in d.groupby("cell"):
    lf=g[g.leakfree]
    x=lf.effect.dropna().values; n=len(x)
    obs=x.mean()
    if n<=20:
        signs=np.array(list(itertools.product([-1,1],repeat=n)))
        null=(signs*x).mean(1); mode="EXACT 2^%d"%n
    else:
        rng=np.random.default_rng(101)
        null=np.array([(rng.choice([-1,1],n)*x).mean() for _ in range(200000)]); mode="200k MC"
    p=float((np.abs(null)>=abs(obs)-1e-15).mean())
    minp=2.0**(1-n) if n<=20 else 5e-6
    print("  %-12s n_blocks=%2d obs=%+7.2f pp  block-level sign-flip p=%.3g (%s, min attainable p=%.2g)"
          % (cell,n,obs*100,p,mode,minp))
print("  NOTE: sign-flip is valid here because the block effect is a paired")
print("  difference and the sharp null makes its sign exchangeable. n=17 gives a")
print("  minimum attainable two-sided p of 2^-16 = 1.5e-5, so n is NOT the binding")
print("  constraint; the test can resolve p far below 0.05.")

print("\n### ATTACK 6a: the frozen bootstrap resamples BLOCKS WITHIN each process but")
print("### NEVER resamples PROCESSES, so between-process variance is excluded from")
print("### every reported CI. This is the real inferential gap; the process-level")
print("### column below closes it at whatever n the frame actually carries.")
print("%-12s %-10s %-24s %-24s" % ("cell","subset","block-boot CI (frozen)","process-level t CI"))
def boot(groups,n_boot=20000,seed=20260830):
    rng=np.random.default_rng(seed); groups=[np.asarray(g,float) for g in groups if len(g)]
    pt=float(np.mean(np.concatenate(groups)))
    dr=np.array([np.mean(np.concatenate([g[rng.integers(0,len(g),len(g))] for g in groups])) for _ in range(n_boot)])
    return pt,float(np.percentile(dr,2.5)),float(np.percentile(dr,97.5))
for cell,g in d.groupby("cell"):
    for name,sub in [("FULL",g),("LEAKFREE",g[g.leakfree])]:
        pt,lo,hi=boot([v.effect.dropna().values for _,v in sub.groupby("replicate")])
        pm=np.array([v.effect.mean() for _,v in sub.groupby("replicate")])
        npr=len(pm); se=pm.std(ddof=1)/np.sqrt(npr); t=stats.t.ppf(0.975,npr-1)
        print("%-12s %-10s %+7.2f [%+7.2f,%+7.2f]   %+7.2f [%+7.2f,%+7.2f]  %s"
              % (cell,name,pt*100,lo*100,hi*100,pm.mean()*100,(pm.mean()-t*se)*100,(pm.mean()+t*se)*100,
                 "EXCLUDES 0" if (pm.mean()-t*se)*(pm.mean()+t*se)>0 else "*** INCLUDES 0 ***"))

print("\n### ATTACK 6b: does the pooled membership-outcome association reduce to")
print("### fault-family MIXING? (leak-free rate differs by family; effect sign differs)")
print(d.groupby("family").agg(leakfree_rate=("leakfree","mean"), mean_effect=("effect","mean")))
lf_o=d[(d.family=='obs')].leakfree.mean(); lf_m=d[(d.family=='motor')].leakfree.mean()
print("  obs blocks are %.1f%% leak-free vs motor %.1f%%; obs effect is +, motor is -." % (lf_o*100,lf_m*100))
print("  => the leak-free pool is obs-enriched, so its pooled mean effect is pulled up.")
print("  Predicted pooled LF-DIV effect gap from mixing alone: %+.2f pp; observed %+.2f pp"
      % (((lf_o*d[d.family=='obs'].effect.mean()/(lf_o+lf_m)*2) - 0)*0 +
         ( (d[d.leakfree].family.eq('obs').mean()-d[~d.leakfree].family.eq('obs').mean())
           *(d[d.family=='obs'].effect.mean()-d[d.family=='motor'].effect.mean()) )*100,
         (d[d.leakfree].effect.mean()-d[~d.leakfree].effect.mean())*100))
