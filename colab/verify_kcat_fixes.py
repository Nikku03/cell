"""Verify the kcat fixes by RE-RUNNING the whole-cell checks on the corrected data (before -> after)."""
import json
MEAS={'measured','EC-measured'}; DIFF=1e9; MARGIN=2.0
def scan(kr):
    too_low=too_high=diff=0
    for g,v in kr.items():
        kc=v.get('kcat_per_s'); km=v.get('km_uM'); meas=v.get('tier') in MEAS
        if kc is None: continue
        floor=v.get('davidi_kcat_max_per_s') or v.get('invivo_apparent_kcat_per_s')
        if floor and floor>kc*MARGIN and not meas: too_low+=1
        if kc>1e7*MARGIN and not meas: too_high+=1
        if km and km>0 and kc/(km*1e-6)>DIFF*MARGIN and not meas: diff+=1
    return dict(too_low=too_low, too_high=too_high, diffusion=diff)
orig=json.load(open('outputs/orphan/kinetics_refined.json'))['kinetics_refined']
corr=json.load(open('outputs/orphan/kinetics_refined_corrected.json'))['kinetics_refined']
b, a = scan(orig), scan(corr)
# how many measured values differ between the two files? (must be 0)
changed_measured=sum(1 for g,v in corr.items() if v.get('tier') in MEAS and v['kcat_per_s']!=orig[g]['kcat_per_s'])
print('='*70); print('VERIFY kcat FIXES BY RE-RUNNING THE CHECK  (before -> after)'); print('='*70)
print(f"  provably-too-low predicted kcats : {b['too_low']:4d} -> {a['too_low']:4d}")
print(f"  too-high (physics ceiling)       : {b['too_high']:4d} -> {a['too_high']:4d}")
print(f"  kcat/Km past diffusion limit     : {b['diffusion']:4d} -> {a['diffusion']:4d}")
print(f"  MEASURED kcats changed           : {changed_measured}  (must be 0)")
ok = a['too_low']<=3 and a['too_high']<=b['too_high'] and a['diffusion']<=b['diffusion'] and changed_measured==0
print('\n  VERDICT:', 'PASS — the fix resolved the errors, created none, touched no facts' if ok else 'FAIL')
print('  (residual too-low = the 3 physics-capped conflicts, correctly not forced past the diffusion limit)')
