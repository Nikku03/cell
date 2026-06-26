"""Can fitness help an organism WITHOUT its own fitness data?
Test on mtub (independent DeJesus truth), pretending mtub has no fitness:
project fitness-essentiality from relatives via feba orthologs, vs conservation.
Result: transferred fitness collapses onto conservation (corr 0.84); on
conservation-blind genes AUC drops 0.717(own) -> 0.558(transferred). Orthogonal
power requires the organism's OWN measurement. See wheel4_nofit_report.md.
"""
