"""Can we COMPUTE [TF] without measuring it? Steady state = production/degradation,
modulated by feedback. Decompose the E. coli TF set by how computable [TF] is:
  gamma (degradation) ~ growth dilution (universal) + degron tags (sequence)
  negative autoreg -> [TF] robust to beta (Alon) -> beta not needed
  + specific operator -> setpoint K also genome-derivable -> [TF] fully computable
  positive autoreg -> bistable -> inherited [TF] from parent picks the attractor
  no autoreg -> [TF]=beta/gamma -> needs beta (promoter strength, NOT computable)
"""
import csv
from collections import defaultdict
SIGN={'activator':1,'repressor':-1}
edges=[]
for l in open('data/external_data/regulon/network_tf_gene.txt'):
    if l.startswith('#') or not l.strip(): continue
    p=l.rstrip().split('\t'); edges.append((p[0].lower(),p[1].lower(),SIGN.get(p[2].lower(),0)))
regs=set(e[0] for e in edges); outdeg=defaultdict(set); auto={}
for tf,tg,s in edges:
    outdeg[tf].add(tg)
    if tg==tf: auto[tf]=s
n=len(regs)
neg=[t for t in regs if auto.get(t)==-1]; pos=[t for t in regs if auto.get(t)==1]
noauto=[t for t in regs if t not in auto]
spec=lambda t: len(outdeg[t])<=30
good=[t for t in neg if spec(t)]
import json
res=dict(total=n, neg_autoreg=len(neg), pos_autoreg=len(pos), no_autoreg=len(noauto),
         neg_and_specific_computable=len(good), pos_and_specific=len([t for t in pos if spec(t)]),
         pct_computable_via_feedback=round(100*(len(good)+len([t for t in pos if spec(t)]))/n))
print(json.dumps(res,indent=2))
json.dump(res,open('outputs/orphan/tf_concentration_feedback.json','w'),indent=2)
