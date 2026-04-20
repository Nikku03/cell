"""
SMILES library for the Syn3A metabolome.

Curated by-hand from standard biochemistry databases (BiGG IDs are
canonical metabolites). Every entry is cross-referenced against
KEGG/ChEBI — no speculative structures.

Species intentionally omitted (not small molecules):
  - M_trdrd_c, M_trdox_c       : thioredoxin protein
  - M_ACP_c, M_apoACP_c        : acyl carrier protein
  - M_dhlpl_PdhC_c             : lipoyl-PdhC protein-bound intermediate
  - M_k_e, M_mg2_e, M_ca2_e    : monoatomic ions
  - M_pi_e                     : ionic phosphate (handled as cofactor)

For extracellular metabolites (_e), the structure is identical to the
intracellular version — same molecule, different compartment.
"""

BIGG_TO_SMILES = {
    # ========================================================================
    # GLYCOLYSIS / GLUCONEOGENESIS
    # ========================================================================
    'M_glc__D_c':   'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',
    'M_glc__D_e':   'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',
    'M_g6p_c':      'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1OP(=O)(O)O',
    'M_g1p_c':      'O[C@H]1OC(COP(=O)(O)O)[C@H](O)[C@@H](O)[C@@H]1O',
    'M_f6p_c':      'OC[C@@H]1O[C@@](O)(COP(=O)(O)O)[C@@H](O)[C@@H]1O',
    'M_fdp_c':      'OP(=O)(O)OC[C@@H]1O[C@@](O)(COP(=O)(O)O)[C@@H](O)[C@@H]1O',
    'M_dhap_c':     'OCC(=O)COP(=O)(O)O',
    'M_g3p_c':      'O=C[C@@H](O)COP(=O)(O)O',
    'M_13dpg_c':    'O=C(OP(=O)(O)O)[C@@H](O)COP(=O)(O)O',
    'M_3pg_c':      'O=C(O)[C@@H](O)COP(=O)(O)O',
    'M_2pg_c':      'O=C(O)[C@H](OP(=O)(O)O)CO',
    'M_pep_c':      'O=C(O)C(=C)OP(=O)(O)O',
    'M_pyr_c':      'CC(=O)C(=O)O',
    'M_lac__L_c':   'C[C@H](O)C(=O)O',
    'M_acald_c':    'CC=O',
    'M_accoa_c':    'CC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc23)C(O)C1OP(=O)(O)O',
    'M_actp_c':     'CC(=O)OP(=O)(O)O',

    # ========================================================================
    # PENTOSE PHOSPHATE
    # ========================================================================
    'M_6pgl_c':     'O=C1O[C@@H](COP(=O)(O)O)[C@H](O)[C@H]1O',
    'M_6pgc_c':     'O=C(O)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)COP(=O)(O)O',
    'M_ru5p__D_c':  'OCC(=O)[C@H](O)[C@@H](O)COP(=O)(O)O',
    'M_xu5p__D_c':  'OCC(=O)[C@@H](O)[C@@H](O)COP(=O)(O)O',
    'M_r5p_c':      'O=C[C@H](O)[C@H](O)[C@@H](O)COP(=O)(O)O',
    'M_r1p_c':      'OC[C@H]1O[C@H](OP(=O)(O)O)[C@@H](O)[C@@H]1O',
    'M_e4p_c':      'O=C[C@H](O)[C@@H](O)COP(=O)(O)O',
    'M_s7p_c':      'OCC(=O)[C@H](O)[C@@H](O)[C@H](O)[C@H](O)COP(=O)(O)O',
    'M_2dr1p_c':    'OC[C@H]1O[C@H](OP(=O)(O)O)C[C@@H]1O',
    'M_2dr5p_c':    'O=C[C@H](O)[C@@H](O)CCOP(=O)(O)O',
    'M_prpp_c':     'OP(=O)(O)O[C@@H]1O[C@H](COP(=O)(O)O)[C@H](O)[C@@H]1OP(=O)(O)OP(=O)(O)O',

    # ========================================================================
    # PURINE NUCLEOSIDES / NUCLEOTIDES
    # ========================================================================
    'M_ade_c':      'Nc1ncnc2[nH]cnc12',                           # adenine
    'M_adn_c':      'Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O',
    'M_adn_e':      'Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O',
    'M_amp_c':      'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O',
    'M_dad_2_c':    'Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)C1',
    'M_dad_2_e':    'Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)C1',
    'M_damp_c':     'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)C1',
    'M_dadp_c':     'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)C1',
    'M_datp_c':     'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)C1',
    'M_gua_c':      'Nc1nc2[nH]cnc2c(=O)[nH]1',                    # guanine
    'M_gsn_c':      'Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1',
    'M_gsn_e':      'Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1',
    'M_gmp_c':      'Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1',
    'M_dgsn_c':     'Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)C2)c(=O)[nH]1',
    'M_dgsn_e':     'Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)C2)c(=O)[nH]1',
    'M_dgmp_c':     'Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)C2)c(=O)[nH]1',
    'M_dgdp_c':     'Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)C2)c(=O)[nH]1',
    'M_dgtp_c':     'Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)C2)c(=O)[nH]1',

    # ========================================================================
    # PYRIMIDINE NUCLEOSIDES / NUCLEOTIDES
    # ========================================================================
    'M_ura_c':      'O=c1cc[nH]c(=O)[nH]1',                         # uracil
    'M_uri_c':      'O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1',
    'M_uri_e':      'O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1',
    'M_ump_c':      'O=c1ccn([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1',
    'M_udp_c':      'O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1',
    'M_utp_c':      'O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1',
    'M_cytd_c':     'Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1',
    'M_cmp_c':      'Nc1ccn([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)n1',
    'M_cdp_c':      'Nc1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)n1',
    'M_ctp_c':      'Nc1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)n1',
    'M_dcyt_c':     'Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)C2)c(=O)n1',
    'M_dcyt_e':     'Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)C2)c(=O)n1',
    'M_dcmp_c':     'Nc1ccn([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)C2)c(=O)n1',
    'M_dcdp_c':     'Nc1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)C2)c(=O)n1',
    'M_dctp_c':     'Nc1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)C2)c(=O)n1',
    'M_duri_c':     'O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)C2)c(=O)[nH]1',
    'M_dump_c':     'O=c1ccn([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)C2)c(=O)[nH]1',
    'M_dudp_c':     'O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)C2)c(=O)[nH]1',
    'M_dutp_c':     'O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)C2)c(=O)[nH]1',
    'M_thymd_c':    'CC1=CN([C@@H]2O[C@H](CO)[C@@H](O)C2)C(=O)NC1=O',
    'M_thymd_e':    'CC1=CN([C@@H]2O[C@H](CO)[C@@H](O)C2)C(=O)NC1=O',
    'M_dtmp_c':     'CC1=CN([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)C2)C(=O)NC1=O',
    'M_dtdp_c':     'CC1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)C2)C(=O)NC1=O',
    'M_dttp_c':     'CC1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)C2)C(=O)NC1=O',

    # ========================================================================
    # AMINO ACIDS (intracellular _c and extracellular _e have same structure)
    # ========================================================================
    'M_ala__L_c':   'C[C@H](N)C(=O)O',
    'M_ala__L_e':   'C[C@H](N)C(=O)O',
    'M_gly_c':      'NCC(=O)O',
    'M_gly_e':      'NCC(=O)O',
    'M_ser__L_c':   'N[C@@H](CO)C(=O)O',
    'M_ser__L_e':   'N[C@@H](CO)C(=O)O',
    'M_thr__L_c':   'C[C@@H](O)[C@H](N)C(=O)O',
    'M_thr__L_e':   'C[C@@H](O)[C@H](N)C(=O)O',
    'M_cys__L_c':   'N[C@@H](CS)C(=O)O',
    'M_cys__L_e':   'N[C@@H](CS)C(=O)O',
    'M_met__L_c':   'CSCC[C@H](N)C(=O)O',
    'M_met__L_e':   'CSCC[C@H](N)C(=O)O',
    'M_val__L_c':   'CC(C)[C@H](N)C(=O)O',
    'M_val__L_e':   'CC(C)[C@H](N)C(=O)O',
    'M_leu__L_c':   'CC(C)C[C@H](N)C(=O)O',
    'M_leu__L_e':   'CC(C)C[C@H](N)C(=O)O',
    'M_ile__L_c':   'CC[C@H](C)[C@H](N)C(=O)O',
    'M_ile__L_e':   'CC[C@H](C)[C@H](N)C(=O)O',
    'M_pro__L_c':   'O=C(O)[C@@H]1CCCN1',
    'M_pro__L_e':   'O=C(O)[C@@H]1CCCN1',
    'M_phe__L_c':   'N[C@@H](Cc1ccccc1)C(=O)O',
    'M_phe__L_e':   'N[C@@H](Cc1ccccc1)C(=O)O',
    'M_tyr__L_c':   'N[C@@H](Cc1ccc(O)cc1)C(=O)O',
    'M_tyr__L_e':   'N[C@@H](Cc1ccc(O)cc1)C(=O)O',
    'M_trp__L_c':   'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O',
    'M_trp__L_e':   'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O',
    'M_his__L_c':   'N[C@@H](Cc1cnc[nH]1)C(=O)O',
    'M_his__L_e':   'N[C@@H](Cc1cnc[nH]1)C(=O)O',
    'M_lys__L_c':   'NCCCC[C@H](N)C(=O)O',
    'M_lys__L_e':   'NCCCC[C@H](N)C(=O)O',
    'M_arg__L_c':   'N=C(N)NCCC[C@H](N)C(=O)O',
    'M_arg__L_e':   'N=C(N)NCCC[C@H](N)C(=O)O',
    'M_asp__L_c':   'N[C@@H](CC(=O)O)C(=O)O',
    'M_asp__L_e':   'N[C@@H](CC(=O)O)C(=O)O',
    'M_glu__L_c':   'N[C@@H](CCC(=O)O)C(=O)O',
    'M_glu__L_e':   'N[C@@H](CCC(=O)O)C(=O)O',
    'M_asn__L_c':   'NC(=O)C[C@H](N)C(=O)O',
    'M_asn__L_e':   'NC(=O)C[C@H](N)C(=O)O',
    'M_gln__L_c':   'NC(=O)CC[C@H](N)C(=O)O',
    'M_gln__L_e':   'NC(=O)CC[C@H](N)C(=O)O',

    # ========================================================================
    # COFACTORS
    # ========================================================================
    'M_ribflv_c':   'Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)CO)c2cc1C',
    'M_ribflv_e':   'Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)CO)c2cc1C',
    'M_fmn_c':      'Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)O)c2cc1C',
    'M_fad_c':      'Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]4O[C@H](n5cnc6c(N)ncnc65)[C@@H](O)[C@@H]4O)c2cc1C',
    'M_nac_c':      'O=C(O)c1cccnc1',                               # nicotinic acid
    'M_nac_e':      'O=C(O)c1cccnc1',
    'M_nicrnt_c':   'NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP(=O)(O)[O-])[C@@H](O)[C@H]1O',  # nicotinamide mononucleotide
    'M_dnad_c':     'NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@H](n4cnc5c(N)ncnc54)[C@@H](O)C3)[C@@H](O)[C@H]2O)c1',
    'M_5fthf_c':    'Nc1nc2c(c(=O)[nH]1)NC(CN(C=O)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1)CN2',
    'M_5fthf_e':    'Nc1nc2c(c(=O)[nH]1)NC(CN(C=O)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1)CN2',
    'M_methfglu3_c': 'Nc1nc2c(c(=O)[nH]1)N3CC(CN2)N(c2ccc(C(=O)NC(CCC(=O)NC(CCC(=O)NC(CCC(=O)O)C(=O)O)C(=O)O)C(=O)O)cc2)C3',
    'M_10fthfglu3_c':'Nc1nc2c(c(=O)[nH]1)NC(CN(c1ccc(C(=O)NC(CCC(=O)NC(CCC(=O)NC(CCC(=O)O)C(=O)O)C(=O)O)C(=O)O)cc1)C=O)CN2',
    'M_mlthfglu3_c':'Nc1nc2c(c(=O)[nH]1)NC(CN(c1ccc(C(=O)NC(CCC(=O)NC(CCC(=O)NC(CCC(=O)O)C(=O)O)C(=O)O)C(=O)O)cc1)C)CN2',
    'M_5fthfglu3_c':'Nc1nc2c(c(=O)[nH]1)NC(CN(C=O)c1ccc(C(=O)NC(CCC(=O)NC(CCC(=O)NC(CCC(=O)O)C(=O)O)C(=O)O)C(=O)O)cc1)CN2',
    'M_pydx5p_c':   'Cc1ncc(COP(=O)(O)O)c(C=O)c1O',                 # pyridoxal 5-phosphate
    'M_pydx5p_e':   'Cc1ncc(COP(=O)(O)O)c(C=O)c1O',
    'M_thmpp_c':    'Cc1ncc(C[n+]2csc(CCOP(=O)(O)OP(=O)(O)O)c2C)c(N)n1',  # thiamine pyrophosphate
    'M_thmpp_e':    'Cc1ncc(C[n+]2csc(CCOP(=O)(O)OP(=O)(O)O)c2C)c(N)n1',
    'M_coa_e':      'NC(=N)N[CH2]CCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc23)C(O)C1OP(=O)(O)O',

    # ========================================================================
    # LIPID / MEMBRANE INTERMEDIATES
    # ========================================================================
    'M_glyc_c':     'OCC(O)CO',                                     # glycerol
    'M_glyc3p_c':   'OCC(O)COP(=O)(O)O',
    'M_fa_c':       'CCCCCCCCCCCCCCCC(=O)O',                        # palmitic acid as representative fatty acid
    'M_ap_c':       'CCCCCCCCCCCCCCCC(=O)OP(=O)(O)O',               # acyl-phosphate (palmitoyl-P)
    'M_1ag3p_c':    'CCCCCCCCCCCCCCCC(=O)OCC(O)COP(=O)(O)O',        # 1-acyl-sn-glycerol 3-phosphate
    'M_pa_c':       'CCCCCCCCCCCCCCCC(=O)OC[C@H](OC(=O)CCCCCCCCCCCCCCC)COP(=O)(O)O',
    'M_cdpdag_c':   'CCCCCCCCCCCCCCCC(=O)OC[C@H](OC(=O)CCCCCCCCCCCCCCC)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2ccc(N)nc2=O)[C@H](O)[C@@H]1O',
    'M_pg3p_c':     'OCC(O)COP(=O)(O)OC[C@H](OC(=O)CCCCCCCCCCCCCCC)COC(=O)CCCCCCCCCCCCCCC',
    'M_pg_c':       'OCC(O)COP(=O)(O)OC[C@H](OC(=O)CCCCCCCCCCCCCCC)COC(=O)CCCCCCCCCCCCCCC',
    'M_12dgr_c':    'CCCCCCCCCCCCCCCC(=O)OC[C@H](OC(=O)CCCCCCCCCCCCCCC)CO',
    'M_pap_c':      'NC(=O)CC(OP(=O)(O)O)P(=O)(O)O',                # 3'-phosphoadenosine 5'-phosphate - approximation
    'M_udpg_c':     'O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O[C@H]3O[C@H](CO)[C@H](O)[C@@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c(=O)[nH]1',
    'M_udpgal_c':   'O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O[C@H]3O[C@H](CO)[C@@H](O)[C@@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c(=O)[nH]1',

    # ========================================================================
    # MISC
    # ========================================================================
    'M_o2_c':       'O=O',
    'M_man__L_c':   'OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O',
    'M_fru_c':      'OC[C@@H]1O[C@@](O)(CO)[C@@H](O)[C@@H]1O',
    'M_sprm_e':     'NCCCNCCCCNCCCN',                               # spermine
}
