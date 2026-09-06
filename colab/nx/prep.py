"""Cache everything the interaction profile needs, once."""
import gzip, json, glob, sys, time, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/user/cell/colab')
SP = Path('/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad')
CACHE = SP / 'nx' / 'bench.pkl'
t0 = time.time()
import nexus_catalyst_esm as M

E, rx, vocab, cats = M.load_bench()
print(f'  esm bench: {len(rx)} reactions, vocab {len(vocab)}, {len(E)} embeddings, dim {len(next(iter(E.values())))}  [{time.time()-t0:.0f}s]')
seq = M.sequences()
print(f'  sequences: {len(seq)}  [{time.time()-t0:.0f}s]')
cat_of = np.array([r['cat'] for r in rx])
cvocab = sorted({r['cat'] for r in rx} | {r['sub'] for r in rx})
fam, ncomp = M.family_of([g for g in cvocab if g in seq], seq)
print(f'  homology clusters: {ncomp}  [{time.time()-t0:.0f}s]')

dock = {}
for f in sorted(glob.glob('/home/user/cell/outputs/orphan/catalyst_pilot_features*.json')):
    dock.update(json.load(open(f)))
print(f'  docked reactions: {len(dock)}')

pickle.dump({'E': E, 'rx': rx, 'vocab': vocab, 'cats': {int(k): v for k, v in cats.items()},
             'seq': {g: seq[g] for g in set(list(cvocab) + vocab) if g in seq},
             'fam': fam, 'ncomp': ncomp, 'dock': dock}, open(CACHE, 'wb'))
print(f'  -> {CACHE}  [{time.time()-t0:.0f}s]')
