"""Bacterial Lifecycle Ontology (BLO) -- functional categorisation
of every gene across all organisms, with branch-level derived
features fed back into the essentiality model.

The framing (from user):
  Bacteria run a fixed life-cycle: maintain genome -> express
  genome -> make energy -> build parts -> wrap themselves ->
  move stuff in/out -> sense -> defend -> interact -> die.
  Each function MUST be present, but different organisms
  implement it with different gene families. So the question
  isn't "is THIS gene essential" but "is the FUNCTION essential,
  and how many alternative solutions exist."

10 life-cycle-ordered branches:
   1. Information maintenance      (DNA repl, repair, division)
   2. Information expression       (txn, txl, RNA processing)
   3. Energy generation             (ATP, ETC, central carbon)
   4. Biosynthesis                  (AA, nt, cofactor, lipid)
   5. Cell envelope                 (PG, membrane, LPS, secretion)
   6. Transport & homeostasis       (pumps, ion, nutrient uptake)
   7. Sensing & regulation          (2CS, sigma factors, stress)
   8. Defense & persistence         (resistance, RM, CRISPR, TA)
   9. Motility & interaction        (flagella, pili, adhesion)
  10. Death & recycling             (autolysins, scavenging)

Pipeline:
  1. classify_by_name(gene_name) -- ~600 well-known E. coli /
       B. subtilis / M. tuberculosis canonical names + prefix
       patterns for big families (rps*, rpl*, fts*, mur*,
       fab*, pur*, pyr*, his*, ilv*, leu*, trp*, bio*, thi*,
       fol*, atp*, nuo*, cyo*, cyd*, kdp*, feo*, lpx*, etc.)
  2. OG-vote propagation: each OG inherits the majority branch
     of its name-classified members, then unclassified members
     of that OG inherit the OG's branch
  3. Per (organism, branch) summary:
       - n_genes, n_essential, frac_essential
       - mean family_frac, mean max_synteny (where joinable)
  4. Two BRANCH-level derived features (THE NEW STUFF):
       - function_essentiality_universal :
           for each branch, fraction of organisms where the
           branch is "operationally essential" (>=40% of the
           branch's genes are essential in that organism).
           High value = function is essential in most organisms
           regardless of which OG implements it.
       - solution_diversity :
           for each branch, number of DISTINCT OG families that
           implement it across the whole tree. High = many
           redundant solutions (rogue-essentials likely).
           Low = single conserved solution (already caught by
           family_frac).
  5. Output blo_features.csv with per-gene branch label +
     those two branch-level features (constant within branch
     but varies across genes).

The hypothesis being tested:
  Rogue-essential FNs at family_frac~=0 should be RECOVERED by
  the model when their branch has high function_essentiality
  AND high solution_diversity -- because the model can learn
  "low family-frac + low solution-diversity = truly dispensable"
  vs. "low family-frac + high solution-diversity + essential
  function = rogue-essential, likely true positive."
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"
LABELS_CSV = DATA_DIR / "labels.csv"
ORTH_CSV   = DATA_DIR / "orthology_features.csv"
SYN_CSV    = DATA_DIR / "synteny_features.csv"
FBA_CSV    = DATA_DIR / "fba_features.csv"
OUT_LONG   = DATA_DIR / "bacterial_lifecycle_per_org.csv"
OUT_FEATS  = DATA_DIR / "blo_features.csv"
OUT_SUMM   = DATA_DIR / "bacterial_lifecycle_summary.json"
OUT_UNCL   = DATA_DIR / "bacterial_lifecycle_unclassified.csv"

# ============================================================
#  Branch definitions: each branch -> rule sets
#    "names"    : exact lowercase gene names
#    "prefixes" : lowercase prefixes (case-insensitive); a gene
#                 matches if its lowercase name starts with the
#                 prefix AND the next char is a letter, digit,
#                 or end-of-string (so "tssA" matches "tss" but
#                 "groEL" does not match "gro" if "gro" not
#                 listed -- we list groE explicitly).
# ============================================================
BLO = [
  ("1_information_maintenance", {
    "1.1_dna_replication": {
        "names": ("dnaa","dnab","dnac","dnae","dnag","dnan","dnaq","dnax",
                  "dnai","dnad","dnat","dnaj","dnak",
                  "pola","polb","polc","gyra","gyrb","topa","topb",
                  "parc","pare","ssb","ssba","ssbb","prim","prima",
                  "hola","holb","holc","hold","hole",
                  "rep","priA","priB","priC","nrdA","nrdB","nrdE","nrdF",
                  "yoaA","tus"),
        "prefixes": ("dnap","poli","polii"),
    },
    "1.2_dna_repair": {
        "names": ("lexa","reca","recb","recc","recd","rece","recf","recg",
                  "recj","recn","reco","recr","rect","recq",
                  "uvra","uvrb","uvrc","uvrd","uvre",
                  "muth","mutl","muts","mutt","muty","mutm","mutd","mutn",
                  "dam","ada","ogt","xtha","xthB","nfo","mfd","ung","fpg",
                  "nei","sbcb","sbcc","sbcd","ding","dinb","dinp","ruva","ruvb","ruvc",
                  "umud","umuc","sula","wrba","yebg","tag","mug","alka","alkb"),
        "prefixes": ("rec","mut","uvr","din","umu","sbc","alk"),
    },
    "1.3_chromosome_partition": {
        "names": ("para","parb","parg","fish","smc","smc1","mukb","muke",
                  "mukf","scpa","scpb","ftsk","sets","spo0j","soj","obg",
                  "mind","mine","minc","mreb","mrec","mred","rodA"),
        "prefixes": (),
    },
    "1.4_cell_division": {
        "names": ("ftsa","ftsb","ftsc","ftsd","ftse","ftsf","ftsh","ftsi",
                  "ftsj","ftsk","ftsl","ftsm","ftsn","ftsp","ftsq","ftsr",
                  "ftss","ftst","ftsu","ftsv","ftsw","ftsx","ftsy","ftsz",
                  "zipa","zapa","zapb","zapc","zapd","damx","pal","tola",
                  "tolq","tolr","tolb","ymgf","ymgg"),
        "prefixes": ("fts","mur","pbp"),
    },
  }),

  ("2_information_expression", {
    "2.1_transcription": {
        "names": ("rpoa","rpob","rpoc","rpod","rpoe","rpoh","rpon","rpos",
                  "rpoz","sigh","sigm","sigb","sigw","sige","sigd","sigy",
                  "siga","sigf","sigg","sigk",
                  "nusa","nusb","nusg","nuse","rho","greA","greB",
                  "fis","hns","ihfa","ihfb","crp","stpA","mfd","hupA","hupB"),
        "prefixes": ("rpo","sig","sigma"),
    },
    "2.2_translation_ribosome": {
        "names": ("efp","tufa","tufb","fusa","tsf",
                  "infa","infb","infc","prfa","prfb","prfc","frr","fmt","def",
                  "rrsa","rrsb","rrsc","rrsd","rrse","rrsg","rrsh",
                  "rrla","rrlb","rrlc","rrld","rrle","rrlg","rrlh",
                  "rrfa","rrfb","rrfc","rrfd","rrfe","rrff","rrfg","rrfh",
                  "trna","trnb"),
        "prefixes": ("rps","rpl","rpm","rrf","rrs","rrl"),
    },
    "2.3_aatrna_synthetases": {
        "names": ("glys","glya","alas","args","asns","asps","cyss","glns",
                  "glus","hiss","iles","leus","lyss","metg","phes","phet",
                  "pros","sers","thrs","trps","tyrs","vals","mets"),
        "prefixes": (),
    },
    "2.4_rna_processing": {
        "names": ("rnc","rnpa","rnpb","rnr","pnp","rne","rng","hfq","rho",
                  "rsma","rsmb","rsmc","rsme","rsmf","rsmg","rsmh","rsmi",
                  "rlma","rlmb","rlmc","rlmd","rlme","rlmf","rlmg","rlmh",
                  "rlmi","rlmj","rlmk","rlml","rlmm","rlmn",
                  "yhgF","yhgG","csra","cspA",
                  "trua","trub","truc","trud"),
        "prefixes": ("rsm","rlm","tru","rim"),
    },
  }),

  ("3_energy_generation", {
    "3.1_atp_synthesis": {
        "names": ("atpa","atpb","atpc","atpd","atpe","atpf","atpg","atph","atpi"),
        "prefixes": ("atp",),
    },
    "3.2_electron_transport": {
        "names": ("nuoa","nuob","nuoc","nuod","nuoe","nuof","nuog","nuoh",
                  "nuoi","nuoj","nuok","nuol","nuom","nuon",
                  "cyoa","cyob","cyoc","cyod","cyoe","cyda","cydb","cydc","cydd",
                  "sdhA","sdhB","sdhC","sdhD","frdA","frdB","frdC","frdD",
                  "ndh","ndha","ndhb","wrba",
                  "ccoo","ccon","ccop","cyca","cycb","cycb6","ccsa","ccsb"),
        "prefixes": ("nuo","cyo","cyd","sdh","frd","cco"),
    },
    "3.3_central_carbon_metab": {
        "names": ("glk","pgi","pfka","pfkb","fbp","fbaA","fbaB","tpia","gapa","pgk",
                  "gpma","eno","pyka","pykf","ppc","pck","ppsa","mdh",
                  "glta","acna","acnb","icd","sucA","sucB","sucC","sucD",
                  "fuma","fumb","fumc","fumar","ace","aceA","aceB","aceE","aceF",
                  "zwf","pgl","gnd","rpia","rpib","tkta","tktb","tala","talb",
                  "ppgK","glpx","ppdK"),
        "prefixes": ("pgi","fba","glt","fum","ace","tkt","tal","pgl"),
    },
  }),

  ("4_biosynthesis", {
    "4.1_amino_acid_biosynth": {
        "names": ("aroa","arob","aroc","arod","aroe","arof","arog","aroh","aroi",
                  "argA","argB","argC","argD","argE","argF","argG","argH","argI","argJ",
                  "metA","metB","metC","metE","metF","metG","metH","metI","metK","metL","metN","metQ","metR",
                  "thrA","thrB","thrC","sera","serB","serc",
                  "lysA","lysC","lysE","lysR","lysS","lysU","lysX","lysY","lysZ",
                  "glnA","glnB","glnD","glnE","glnK","gltA","gltB","gltD","gltX",
                  "ilvA","ilvB","ilvC","ilvD","ilvE","ilvG","ilvH","ilvI","ilvL","ilvM","ilvN","ilvO",
                  "leuA","leuB","leuC","leuD",
                  "trpA","trpB","trpC","trpD","trpE","trpR","trpS",
                  "tyrA","tyrB","tyrR","pheA","pheL","pheR","aspA","aspB","aspC",
                  "proA","proB","proC","prov","prox","proy",
                  "hisA","hisB","hisC","hisD","hisE","hisF","hisG","hisH","hisI",
                  "cysE","cysG","cysH","cysI","cysJ","cysK","cysM","cysN","cysO","cysP","cysQ","cysS","cysU","cysW","cysX","cysZ",
                  "asnA","asnB","glya","glyb","glyc","glyq","glys"),
        "prefixes": ("his","ilv","leu","trp","arg","met","lys","ser","thr","cys","pro","tyr","phe","asp","asn","glt","gln","aro"),
    },
    "4.2_nucleotide_biosynth": {
        "names": ("purA","purB","purC","purD","purE","purF","purH","purK","purL","purM","purN","purP","purR","purS","purT","purU",
                  "pyrB","pyrC","pyrD","pyrE","pyrF","pyrG","pyrH","pyrI",
                  "gua","guaA","guaB","guaC","guaD",
                  "nrdA","nrdB","nrdC","nrdD","nrdE","nrdF","nrdG","nrdH","nrdI","nrdJ",
                  "dut","thya","tdk","ndk","udp","upp","udk","cmk","gmk","tmk",
                  "ribf","apt","gpt","hpt","cdd","add","deoA","deoB","deoC","deoD"),
        "prefixes": ("pur","pyr","nrd"),
    },
    "4.3_cofactor_vitamin": {
        "names": ("ribA","ribB","ribC","ribD","ribE","ribF","ribG","ribH",
                  "folA","folB","folC","folD","folE","folK","folM","folP","folX",
                  "panB","panC","panD","panE","panK","panP","panZ",
                  "thiC","thiD","thiE","thiF","thiG","thiH","thiI","thiK","thiL","thiM","thiN","thiO","thiP","thiQ","thiS",
                  "bioA","bioB","bioC","bioD","bioF","bioG","bioH",
                  "pdxA","pdxB","pdxH","pdxJ","pdxK","pdxY","pdxN",
                  "cobA","cobB","cobC","cobD","cobE","cobF","cobG","cobH","cobI","cobJ","cobK","cobL","cobM","cobN","cobO","cobP","cobQ","cobR","cobS","cobT","cobU","cobV","cobW","cobX","cobY","cobZ",
                  "hemA","hemB","hemC","hemD","hemE","hemF","hemG","hemH","hemL","hemN","hemQ","hemX","hemY",
                  "moaA","moaB","moaC","moaD","moaE","moaF","mob","mobA","mobB",
                  "ispA","ispB","ispC","ispD","ispE","ispF","ispG","ispH","dxs","dxr",
                  "menA","menB","menC","menD","menE","menF","menG","menH","menI",
                  "ubiA","ubiB","ubiC","ubiD","ubiE","ubiF","ubiG","ubiH","ubiI","ubiX",
                  "lipA","lipB","lipC","lipM"),
        "prefixes": ("rib","fol","pan","thi","bio","pdx","cob","hem","moa","mob","isp","men","ubi","lip","ndh","coa"),
    },
    "4.4_lipid_biosynth": {
        "names": ("fabA","fabB","fabD","fabF","fabG","fabH","fabI","fabL","fabM","fabZ",
                  "plsB","plsC","plsX","plsY","pgsA","cdsA","cls",
                  "lpd","accA","accB","accC","accD","accE","accH","fadA","fadB",
                  "yfcZ","yibN"),
        "prefixes": ("fab","fad","acc","pls","des","mvA","mvK"),
    },
  }),

  ("5_cell_envelope", {
    "5.1_peptidoglycan_wall": {
        "names": ("murA","murB","murC","murD","murE","murF","murG","murI","murJ",
                  "ddl","ddlA","ddlB","glmS","glmU","glmM","mraY","mraz","mrcA","mrcB","mreB","mreC","mreD",
                  "dapA","dapB","dapD","dapE","dapF","lysA","alr","dadA","dadX",
                  "bacA","amiA","amiB","amiC","amiD","amiE","amiF","ldcA","mepA","mepM","mepS","mepW","yebA","yifM",
                  "pbpA","pbpB","pbpC","pbpD","pbpE","pbpF","pbpG","pbp1","pbp2","pbp3","pbp4","pbp5","pbp6","pbp7"),
        "prefixes": ("mur","mre","dap","pbp","ami","alr","mep","spo"),
    },
    "5.2_membrane_biosynth_repair": {
        "names": ("plsB","plsC","plsX","plsY","pgsA","cdsA","cls","yfdY",
                  "yhcB","yiaW","ybeT","yejL","ydaJ","yfkA","yfgM"),
        "prefixes": ("yqf","yiC","yia"),
    },
    "5.3_lps_outer_membrane": {
        "names": ("lpxA","lpxB","lpxC","lpxD","lpxH","lpxK","lpxL","lpxM","lpxO","lpxP","lpxR","lpxS","lpxT","lpxX",
                  "waaA","waaB","waaC","waaD","waaE","waaF","waaG","waaH","waaI","waaJ","waaK","waaL","waaM","waaN","waaO","waaP","waaQ","waaR","waaS","waaT","waaU","waaV","waaW","waaX","waaY","waaZ",
                  "rfaA","rfaB","rfaC","rfaD","rfaE","rfaF","rfaG","rfaH","rfaI","rfaJ","rfaK","rfaL","rfaP","rfaQ","rfaY","rfaZ",
                  "kdsA","kdsB","kdsC","kdsD","gmha","gmhB",
                  "ompA","ompC","ompF","ompG","ompL","ompT","ompW","ompX","tolC","lamb","phoE","mtgA","mtgB",
                  "lpp","lolA","lolB","lolC","lolD","lolE",
                  "bamA","bamB","bamC","bamD","bamE","yaeT","yfgM","yfiO","yidC",
                  "msbA","mlaA","mlaB","mlaC","mlaD","mlaE","mlaF","pldA"),
        "prefixes": ("lpx","waa","rfa","kds","gmh","omp","tol","lol","bam","mla"),
    },
    "5.4_secretion_machinery": {
        "names": ("secA","secB","secD","secE","secF","secG","secM","secY","yajC","yidC","ffh","ftsy",
                  "tatA","tatB","tatC","tatD","tatE",
                  "hlyA","hlyB","hlyC","hlyD","tolC",
                  "gspC","gspD","gspE","gspF","gspG","gspH","gspI","gspJ","gspK","gspL","gspM","gspN","gspO",
                  "yscC","yscD","yscE","yscF","yscG","yscH","yscI","yscJ","yscK","yscL","yscM","yscN","yscO","yscP","yscQ","yscR","yscS","yscT","yscU","yscV","yscW","yscX","yscY",
                  "tssA","tssB","tssC","tssD","tssE","tssF","tssG","tssH","tssI","tssJ","tssK","tssL","tssM","hcp","vgrG",
                  "virB1","virB2","virB3","virB4","virB5","virB6","virB7","virB8","virB9","virB10","virB11"),
        "prefixes": ("sec","tat","gsp","ysc","yst","tss","virB","virD","cag"),
    },
  }),

  ("6_transport_homeostasis", {
    "6.1_abc_transporters": {
        "names": ("artJ","artM","artP","artQ","artI","artR",
                  "rbsA","rbsB","rbsC","rbsD","rbsK","rbsR",
                  "cysA","cysP","cysU","cysW","cyse",
                  "ugpA","ugpB","ugpC","ugpE","ugpQ",
                  "malE","malF","malG","malK","malM","malT","mall","malp","malq","mals","malz",
                  "potA","potB","potC","potD","potE","potF","potG","potH","potI",
                  "phnC","phnD","phnE","phnF","phnG","phnH","phnI","phnJ","phnK","phnL","phnM","phnN","phnO","phnP","phnR","phnS","phnT","phnU","phnV","phnW","phnX",
                  "yejA","yejB","yejE","yejF","mdla","mdlb","yhdW","yhdX","yhdY","yhdZ",
                  "oppA","oppB","oppC","oppD","oppF","dppA","dppB","dppC","dppD","dppF",
                  "nikA","nikB","nikC","nikD","nikE",
                  "modA","modB","modC","modE","modF"),
        "prefixes": ("art","rbs","ugp","mal","pot","phn","opp","dpp","nik","mod","liv","glu","oli"),
    },
    "6.2_pts_sugar_uptake": {
        "names": ("ptsH","ptsI","ptsG","ptsM","ptsN","ptsP","ptsO","ptsA","ptsP","crr","fpr","fruR","fruA","fruB","fruF","fruK",
                  "manX","manY","manZ","mtlA","mtlD","mtlF","mtlR","srlA","srlB","srlD","srlE","srlR","srlM","gatA","gatB","gatC","gatD","gatY","gatZ","gatR",
                  "nagA","nagB","nagC","nagD","nagE","nagK","nagZ"),
        "prefixes": ("pts","man","mtl","srl","gat","nag","scr","bgl","tre","lac","gal","glv","mlc"),
    },
    "6.3_efflux_resistance_pumps": {
        "names": ("acrA","acrB","acrD","acrE","acrF","acrR","acrZ","tolC",
                  "mexA","mexB","mexC","mexD","mexE","mexF","mexG","mexH","mexI","mexJ","mexK","mexL","mexM","mexN","mexO","mexP","mexQ","mexR","mexS","mexT","mexU","mexV","mexW","mexX","mexY","mexZ","oprM","oprN","oprJ","oprD","oprK","oprH","oprB","oprC","oprE","oprF","oprG",
                  "emrA","emrB","emrC","emrD","emrE","emrK","emrR","emrY","mdfa","mdtA","mdtB","mdtC","mdtD","mdtE","mdtF","mdtG","mdtH","mdtI","mdtJ","mdtK","mdtL","mdtM","mdtN","mdtO","mdtP",
                  "norA","norB","norC","norG","ydgF","ydgE"),
        "prefixes": ("acr","mex","emr","mdt","nor","mdf","opr","mfs"),
    },
    "6.4_ion_homeostasis": {
        "names": ("kdpA","kdpB","kdpC","kdpD","kdpE","kdpF","kdpG",
                  "trkA","trkB","trkG","trkH","trkE","kup","kefA","kefB","kefC","kefF","kefG",
                  "mgtA","mgtB","mgtC","mgtE","corA","corB","corC","mntA","mntB","mntC","mntD","mntE","mntH","mntP","mntR","mntS",
                  "nhaA","nhaB","nhaC","nhaR","mdfA","sapA","sapB","sapC","sapD","sapF","yejA","ydeI",
                  "zntA","zntB","zntR","zinT","zur","zupT","znuA","znuB","znuC","znuD","ybdA",
                  "copA","copB","copC","copD","copR","copS","copY","cueO","cueR","cusA","cusB","cusC","cusD","cusE","cusF","cusR","cusS"),
        "prefixes": ("kdp","trk","kef","mgt","cor","mnt","nha","znt","znu","cop","cus","mgo"),
    },
    "6.5_iron_uptake": {
        "names": ("feoA","feoB","feoC","fepA","fepB","fepC","fepD","fepE","fepG",
                  "fhuA","fhuB","fhuC","fhuD","fhuE","fhuF",
                  "fecA","fecB","fecC","fecD","fecE","fecI","fecR",
                  "fhuA","entA","entB","entC","entD","entE","entF","entH","entS","ybdZ","exbB","exbD","tonB",
                  "iro","irp","yer","mbtA","mbtB","mbtC","mbtD","mbtE","mbtF","mbtG","mbtH","mbtI","mbtJ","mbtK"),
        "prefixes": ("feo","fep","fhu","fec","ent","mbt","irp","yfi","afu","iuc","iut"),
    },
  }),

  ("7_sensing_regulation", {
    "7.1_two_component_systems": {
        "names": ("phoB","phoR","phoP","phoQ","phoU","phoH","phoE","ompR","envZ",
                  "cpxA","cpxR","cpxP","rcsA","rcsB","rcsC","rcsD","rcsF",
                  "arcA","arcB","baeS","baeR","barA","uvrY",
                  "narP","narQ","narX","narL","creB","creC","cusS","cusR","kdpD","kdpE",
                  "nrgA","nrgB","glnG","glnL","glnK","nac",
                  "atoS","atoC","cheA","cheB","cheR","cheY","cheZ","cheV","cheW","cheC","cheD",
                  "qseB","qseC","ttrR","ttrS","yedV","yedW","yfhA","yfhK","yedW",
                  "agrA","agrB","agrC","agrD","vncR","vncS","vraR","vraS","vraT","tcr","saes","saer","walK","walR","yycF","yycG"),
        "prefixes": ("agr","wal","sae","yyc","vra","yfh","yed","yhj","yhi","yhc","ypd","ypj","ypa","yco","yfk","ymo","spa","tcs"),
    },
    "7.2_stress_response": {
        "names": ("groE","groL","groS","groEL","groES","dnaK","dnaJ","grpE",
                  "clpA","clpB","clpC","clpE","clpL","clpP","clpQ","clpS","clpV","clpX","clpY",
                  "hslU","hslV","hslO","hslR","hflB","hflC","hflK","hflX","ftsh",
                  "sodA","sodB","sodC","sodM","sodF","katE","katG","katA","katB",
                  "oxyR","oxyS","ahpC","ahpF","dps","msrA","msrB","msrC","msrQ","gor","grxA","grxB","grxC","grxD","trxA","trxB","trxC",
                  "cspA","cspB","cspC","cspD","cspE","cspF","cspG","cspH","cspI",
                  "uspA","uspB","uspC","uspD","uspE","uspF","uspG",
                  "relA","spoT","ppGpp","dksA","ggap","csrA","csrB","csrC","csrD",
                  "rpoH","rpoE","rpoS","rseA","rseB","rseC"),
        "prefixes": ("clp","hfl","sod","kat","msr","grx","trx","csp","usp","rsb","yvy","rpoH"),
    },
    "7.3_quorum_signaling": {
        "names": ("luxS","luxI","luxR","luxA","luxB","luxC","luxD","luxE","luxF","luxG","luxN","luxQ","luxO","luxP","luxS","luxT","luxU","luxV","aiia",
                  "lsrA","lsrB","lsrC","lsrD","lsrE","lsrF","lsrG","lsrK","lsrR",
                  "ggA","ggE","ggH","ggI","ggK","yhdA","ydeH","ydiV","yedQ","yhjH","yfiB","yfiN","yfiR","yegE","yhjK","yliE","ycgF","ycgG","ydaM","yneF",
                  "comA","comB","comC","comD","comE","comF","comG","comK","comX","comZ"),
        "prefixes": ("lux","lsr","com","gg","pde"),
    },
  }),

  ("8_defense_persistence", {
    "8.1_phage_defense": {
        "names": ("hsdR","hsdM","hsdS","dpnA","dpnB","dpnC","dpnD","mcrA","mcrB","mcrC","mcrD","mrr","yhdJ","ysdC","tagA",
                  "cas1","cas2","cas3","cas4","cas5","cas6","cas7","cas8","cas9","cas10","cas11","cas12","cas13",
                  "cmrA","cmrB","cmrC","cmrD","cmrE","cmrF","csmA","csmB","csmC","csmD","csmE","csmF","csyA","csyB","csyC","csyD","cse1","cse2","cse3","cse4","csn1","csn2","csa1","csa2","csa3","csa4","csa5"),
        "prefixes": ("hsd","dpn","mcr","cas","cmr","csm","csy","cse","csn","csa","cst","csb","csd","csf","csh","csx"),
    },
    "8.2_toxin_antitoxin": {
        "names": ("hipA","hipB","mazE","mazF","relB","relE","ccdA","ccdB","parD","parE","vapB","vapC","ndoA","ndoI","hicA","hicB","yefM","yoeB","mqsA","mqsR","yafQ","dinJ","yhaV","prlF",
                  "yefM","yoeB","higA","higB","higB1","higB2","higB3","higB4","higB5","kid","kis","epsilon","zeta","phd","doc","atu"),
        "prefixes": ("hip","maz","ccd","par","vap","ndo","hic","yef","yoe","mqs","yaf","din","yha","prl","hig","kid"),
    },
    "8.3_antibiotic_resistance": {
        "names": ("blaCTX","blaSHV","blaTEM","blaOXA","blaKPC","blaNDM","blaIMP","blaVIM","blaCMY","blaACT","blaACC","blaDHA","blaFOX","blaMOX","blaLAT",
                  "mecA","mecR","mecI","vanA","vanB","vanC","vanD","vanE","vanG","vanH","vanR","vanS","vanW","vanX","vanY","vanZ",
                  "ampC","ampD","ampE","ampF","ampG","ampH","ampR","aac","aad","aph","str","tet","cat","ery","mph","mef","msr","sat","dfr","sul",
                  "qnrA","qnrB","qnrC","qnrD","qnrE","qnrS","qep","oqxA","oqxB"),
        "prefixes": ("bla","van","tet","cat","ery","aac","aad","aph","mph","qnr","oqx","mef","ant","dfr","sul"),
    },
  }),

  ("9_motility_interaction", {
    "9.1_flagella_motility": {
        "names": ("flgA","flgB","flgC","flgD","flgE","flgF","flgG","flgH","flgI","flgJ","flgK","flgL","flgM","flgN","flgK",
                  "fliA","fliB","fliC","fliD","fliE","fliF","fliG","fliH","fliI","fliJ","fliK","fliL","fliM","fliN","fliO","fliP","fliQ","fliR","fliS","fliT","fliY","fliZ",
                  "flhA","flhB","flhC","flhD","flhE","flhF","motA","motB","motC","motD","motX","motY"),
        "prefixes": ("flg","fli","flh","mot"),
    },
    "9.2_chemotaxis": {
        "names": ("cheA","cheB","cheR","cheW","cheV","cheY","cheZ","cheD","cheC",
                  "tar","tsr","trg","tap","aer","mcp","frzA","frzB","frzC","frzCD","frzE","frzF","frzG","frzZ"),
        "prefixes": ("che","frz","mcp"),
    },
    "9.3_pili_adhesion": {
        "names": ("pilA","pilB","pilC","pilD","pilE","pilF","pilG","pilH","pilI","pilJ","pilK","pilL","pilM","pilN","pilO","pilP","pilQ","pilR","pilS","pilT","pilU","pilV","pilW","pilX","pilY","pilZ",
                  "fimA","fimB","fimC","fimD","fimE","fimF","fimG","fimH","fimI",
                  "papA","papB","papC","papD","papE","papF","papG","papH","papI","papJ","papK",
                  "csgA","csgB","csgC","csgD","csgE","csgF","csgG","yhaI","yhaJ","yegB","yfaN","yqi"),
        "prefixes": ("pil","fim","pap","csg","afa","sef","saf","stb","fae"),
    },
    "9.4_biofilm": {
        "names": ("bsmA","bsmB","bdcA","cabA","cabB","csgA","csgB","csgC","csgD","csgE","csgF","csgG","pgaA","pgaB","pgaC","pgaD","wcaA","wcaB","wcaC","wcaD","wcaE","wcaF","wcaG","wcaH","wcaI","wcaJ","wcaK","wcaL","wcaM","wzc","wza","wzb",
                  "vpsA","vpsB","vpsC","vpsD","vpsE","vpsF","vpsG","vpsH","vpsI","vpsJ","vpsK","vpsL","vpsM","vpsN","vpsO","vpsP","vpsR","vpsT","vpsU"),
        "prefixes": ("vps","pga","wca","ica","eps","wzx","wzy","wzz","epsA"),
    },
  }),

  ("10_death_recycling", {
    "10.1_autolysins_pcd": {
        "names": ("lytA","lytB","lytC","lytD","lytE","lytF","lytG","lytH","lytM","lytN","lytR","lytS","lytT",
                  "atlA","atlB","atlC","atlD","atlE","atlL","atlW","atlX",
                  "sleA","sleB","sleC","cwlA","cwlB","cwlC","cwlD","cwlE","cwlF","cwlG","cwlH","cwlI","cwlJ","cwlK","cwlL","cwlM","cwlO","cwlP","cwlQ","cwlR","cwlS","cwlT","cwlU","cwlV",
                  "spoIID","spoIIE","spoIIM","spoIIP","spoIIQ","spoIIR","spoIIIE",
                  "ami","aim","lrgA","lrgB","cidA","cidB","cidC",
                  "yfeA","ywbG"),
        "prefixes": ("lyt","atl","sle","cwl","spm","ywp","yvc","ywf","ywh","ywp","lrg","cid"),
    },
  }),
]

# Verbose-name keyword matches (BERIL & many RefSeq annotations use
# descriptive strings rather than 3-4 letter symbols). Order within
# each tuple: more-specific first (longer phrases match before shorter).
# Each phrase is matched as a *substring* of the lower-cased gene_name.
BLO_KEYWORDS = {
  "1.1_dna_replication": (
      "dna polymerase iii","dna polymerase","dna primase","primosome",
      "dna gyrase","gyrase subunit","topoisomerase",
      "replicative helicase","replication initiator",
      "single-strand dna","single-stranded dna-binding",
      "clamp loader","ribonucleotide reductase",
  ),
  "1.2_dna_repair": (
      "mismatch repair","nucleotide excision repair","base excision repair",
      "dna repair","dna recombinase","reca",
      "uvrabc","endonuclease iv","exonuclease iii",
      "dna glycosylase","photolyase","ada regulatory",
      "sos response","lexa","umuc","umud",
  ),
  "1.3_chromosome_partition": (
      "chromosome partition","chromosome segregation","condensin","smc",
      "parab partition","muk subunit",
  ),
  "1.4_cell_division": (
      "cell division protein","septation","septum formation",
      "ftsz","penicillin-binding protein","penicillin binding",
      "min system","murein hydrolase regulator",
  ),
  "2.1_transcription": (
      "rna polymerase","dna-directed rna polymerase","sigma factor",
      "sigma-70","sigma-32","sigma-54","sigma-e","sigma-b",
      "transcription antitermination","transcription elongation",
      "transcription termination","nusa","nusb",
  ),
  "2.2_translation_ribosome": (
      "ribosomal protein","30s ribosomal","50s ribosomal",
      "elongation factor","initiation factor","release factor",
      "ribosome","16s rrna","23s rrna","5s rrna",
      "ribosomal rna","rrna methyltransferase","peptide chain release",
      "ribosome-recycling factor","trigger factor",
  ),
  "2.3_aatrna_synthetases": (
      "aminoacyl-trna","trna synthetase","trna ligase",
      "aminoacyl trna","tyrosyl-trna","leucyl-trna","valyl-trna",
      "isoleucyl-trna","seryl-trna","methionyl-trna","arginyl-trna",
      "alanyl-trna","glycyl-trna","threonyl-trna","prolyl-trna",
      "histidyl-trna","aspartyl-trna","asparaginyl-trna",
      "glutamyl-trna","glutaminyl-trna","cysteinyl-trna","lysyl-trna",
      "phenylalanyl-trna","tryptophanyl-trna",
  ),
  "2.4_rna_processing": (
      "ribonuclease","rnase","rna helicase","rna methyltransferase",
      "pseudouridine synthase","trna pseudouridine","trna modification",
      "rrna pseudouridine","polynucleotide phosphorylase",
  ),
  "3.1_atp_synthesis": (
      "atp synthase","f0f1","f1f0","f-type atpase",
      "atp synthase subunit","proton-translocating atpase",
  ),
  "3.2_electron_transport": (
      "nadh dehydrogenase","nadh:quinone","nadh-quinone",
      "cytochrome c","cytochrome bd","cytochrome bo",
      "cytochrome b6","cytochrome o","cytochrome oxidase","ubiquinol oxidase",
      "succinate dehydrogenase","fumarate reductase","quinol oxidase",
      "ubiquinone","menaquinone biosynth","plastoquinone",
  ),
  "3.3_central_carbon_metab": (
      "glyceraldehyde-3-phosphate","phosphoglycerate kinase","enolase",
      "pyruvate kinase","phosphofructokinase","fructose-bisphosphate",
      "triosephosphate isomerase","pyruvate dehydrogenase",
      "citrate synthase","aconitase","aconitate hydratase",
      "isocitrate dehydrogenase","2-oxoglutarate","alpha-ketoglutarate",
      "succinyl-coa","fumarase","fumarate hydratase","malate dehydrogenase",
      "transketolase","transaldolase","ribulose-phosphate",
  ),
  "4.1_amino_acid_biosynth": (
      "amino acid biosynthesis","aminotransferase","amino-acid biosynthesis",
      "shikimate","chorismate","prephenate","anthranilate",
      "histidinol","imidazoleglycerol","threonine synthase",
      "homoserine","cystathionine","methionine biosynthesis",
      "serine biosynthesis","glutamate synthase","glutamine synthetase",
      "asparagine synthetase","ornithine","argininosuccinate",
      "diaminopimelate","aspartate kinase",
      "indole-3-glycerol","tryptophan synthase","prephenate dehydrogenase",
      "branched-chain amino","leucine biosynthesis","valine biosynthesis",
  ),
  "4.2_nucleotide_biosynth": (
      "purine biosynthesis","pyrimidine biosynthesis","nucleotide biosynthesis",
      "ribonucleotide reductase","thymidylate synthase","dihydroorotate",
      "carbamoyl-phosphate","orotidine","adenylosuccinate",
      "amidophosphoribosyltransferase","phosphoribosyl",
      "thymidine kinase","thymidylate kinase","dutpase",
      "nucleoside diphosphate kinase",
  ),
  "4.3_cofactor_vitamin": (
      "biotin biosynth","biotin synthase","thiamine biosynth","thiamin biosynth",
      "thiamine pyrophosphate","riboflavin biosynth","fmn ","fad ",
      "folate biosynth","dihydrofolate","dihydropteroate","tetrahydrofolate",
      "pantothenate","coenzyme a biosynth","pyridoxal","pyridoxine",
      "cobalamin","vitamin b12","heme biosynth","porphyrin",
      "molybdopterin","molybdenum cofactor",
      "lipoyl","lipoate","menaquinone biosynth","ubiquinone biosynth",
      "isoprenoid biosynth","octaprenyl",
  ),
  "4.4_lipid_biosynth": (
      "fatty acid biosynth","fatty acid synthesis","acyl carrier",
      "fatty-acid synthase","beta-ketoacyl","enoyl-acyl-carrier",
      "acetyl-coa carboxylase","glycerol-3-phosphate dehydrogenase",
      "phosphatidylserine","phosphatidylethanolamine","cardiolipin",
      "lipid a biosynth","cdp-diacylglycerol",
  ),
  "5.1_peptidoglycan_wall": (
      "peptidoglycan","murein","cell wall biosynth","cell wall synthesis",
      "udp-n-acetyl","d-alanyl-d-alanine","d-alanine--d-alanine",
      "muramidase","muramoyl","cell wall hydrolase","peptidoglycan glycosyltransf",
  ),
  "5.2_membrane_biosynth_repair": (
      "lipid metabolism","membrane lipid","lipoprotein biosynth",
      "phosphatidate","cardiolipin synthase","membrane-bound lytic",
  ),
  "5.3_lps_outer_membrane": (
      "lipopolysaccharide","lipid a","kdo","outer membrane",
      "outer membrane protein","porin","bam complex","beta-barrel",
      "lol system","murein lipoprotein","mla pathway","periplasmic",
      "o-antigen","lps biosynth","heptosyltransferase","kds",
  ),
  "5.4_secretion_machinery": (
      "preprotein translocase","sec-dependent","sec translocase",
      "twin-arginine translocation","tat system",
      "type i secretion","type ii secretion","type iii secretion",
      "type iv secretion","type v secretion","type vi secretion",
      "secretin","general secretion","signal peptidase",
  ),
  "6.1_abc_transporters": (
      "abc transporter","abc-type","atp-binding cassette","atp-binding protein",
      "substrate-binding protein","periplasmic-binding protein",
      "permease protein","membrane component",
      "binding-protein-dependent transport","oligopeptide-binding",
      "dipeptide-binding","polar amino acid",
  ),
  "6.2_pts_sugar_uptake": (
      "phosphotransferase system","pts system","sugar-specific iia",
      "iib component","iic component","enzyme i ",
      "enzyme ii ","sugar phosphate","sugar:proton symporter",
  ),
  "6.3_efflux_resistance_pumps": (
      "efflux pump","multidrug","multi-drug","drug efflux",
      "mfs transporter","major facilitator","rnd efflux","rnd family",
      "drug:proton antiporter","mate family",
      "outer membrane efflux","outer-membrane efflux",
  ),
  "6.4_ion_homeostasis": (
      "potassium uptake","potassium transport","potassium channel",
      "magnesium transport","sodium/proton antiporter","na+/h+ antiporter",
      "calcium transport","zinc uptake","zinc transport","copper-translocating",
      "manganese transport","cation transport","cation efflux",
      "heavy metal","arsenate","arsenite",
  ),
  "6.5_iron_uptake": (
      "ferric iron","ferrous iron","iron transport","iron uptake",
      "siderophore","heme transport","heme uptake","ton system",
      "tonb-dependent","enterobactin","enterochelin",
      "iron-regulated","high-affinity iron",
  ),
  "7.1_two_component_systems": (
      "histidine kinase","sensor histidine","sensor kinase",
      "response regulator","two-component system","two component",
      "winged helix-turn-helix",
  ),
  "7.2_stress_response": (
      "heat shock protein","chaperone","chaperonin","co-chaperonin",
      "molecular chaperone","cold-shock","cold shock",
      "stringent response","ppgpp","alarmone","spot",
      "universal stress","peroxidase","superoxide dismutase","catalase",
      "glutathione","thioredoxin","glutaredoxin","alkyl hydroperoxide",
      "atp-dependent clp","atp-dependent protease",
      "lon protease","hsl protease",
  ),
  "7.3_quorum_signaling": (
      "quorum sensing","autoinducer","cyclic di-gmp","c-di-gmp",
      "diguanylate cyclase","phosphodiesterase",
      "competence","comp ",
  ),
  "8.1_phage_defense": (
      "restriction enzyme","restriction-modification","type i restriction",
      "type ii restriction","modification methylase","dna methyltransferase",
      "crispr-associated","crispr","cas protein",
      "anti-phage","phage shock protein",
  ),
  "8.2_toxin_antitoxin": (
      "toxin-antitoxin","antitoxin","mazf","relbe","plasmid stability",
      "addiction module","mrna interferase",
  ),
  "8.3_antibiotic_resistance": (
      "beta-lactamase","penicillin amidase","aminoglycoside",
      "tetracycline resistance","chloramphenicol acetyltransferase",
      "erythromycin","macrolide","fluoroquinolone resistance",
      "vancomycin resistance","methicillin",
  ),
  "9.1_flagella_motility": (
      "flagellar","flagellum","flagellin","flagell-",
      "flagellar motor","motility protein","hook ","hook-associated",
  ),
  "9.2_chemotaxis": (
      "chemotaxis","methyl-accepting chemotaxis","chemoreceptor",
      "chea ","chey ","cheb ","cher ","chez ",
  ),
  "9.3_pili_adhesion": (
      "pilus assembly","pilin","fimbrial","fimbriae","type iv pilus",
      "adhesin","autotransporter","trimeric autotransporter",
      "curli","csg",
  ),
  "9.4_biofilm": (
      "biofilm","exopolysaccharide","capsular polysaccharide",
      "extracellular polysaccharide","alginate biosynth",
      "cellulose biosynth","vps","pga",
  ),
  "10.1_autolysins_pcd": (
      "autolysin","n-acetylmuramoyl-l-alanine amidase","lytic transglycosylase",
      "cell wall amidase","peptidoglycan hydrolase",
      "n-acetylglucosaminidase","murein hydrolase","muramidase",
      "endolysin","holin",
  ),
}

# Flatten into a lookup: (lc_name -> sub_branch_id), prefix list,
# keyword list (substring matches).
_NAME_LOOKUP: dict[str, str] = {}
_PREFIX_RULES: list[tuple[str, str]] = []  # (prefix, sub_id), longest first
_KEYWORD_RULES: list[tuple[str, str]] = []  # (keyword, sub_id), longest first
_BRANCH_OF_SUB: dict[str, str] = {}  # sub_id -> top branch id

for branch_id, subs in BLO:
    for sub_id, rules in subs.items():
        _BRANCH_OF_SUB[sub_id] = branch_id
        for n in rules["names"]:
            _NAME_LOOKUP[n.lower()] = sub_id
        for p in rules["prefixes"]:
            _PREFIX_RULES.append((p.lower(), sub_id))

for sub_id, keywords in BLO_KEYWORDS.items():
    if sub_id not in _BRANCH_OF_SUB:
        # keyword bucket for an unknown sub-branch -- skip
        continue
    for kw in keywords:
        _KEYWORD_RULES.append((kw.lower(), sub_id))

_PREFIX_RULES.sort(key=lambda kv: -len(kv[0]))   # match longest first
_KEYWORD_RULES.sort(key=lambda kv: -len(kv[0]))


def classify_by_name(gene_name) -> str | None:
    """Classify a gene to a sub-branch by name. Returns sub_id or None.

    Three-stage matching:
      1. Exact lowercase name match (canonical symbols: 'dnaA', 'rpsA')
      2. Prefix match for big families ('rps' -> ribosomal proteins)
      3. Substring keyword match for verbose descriptors
         ('ribosomal protein S12' -> translation; 'ABC transporter
         ATP-binding protein' -> ABC transport)
    """
    if gene_name is None: return None
    s = str(gene_name).strip().lower()
    if not s or s in {"nan", "none", "-", ""}: return None
    # 1. Exact match
    if s in _NAME_LOOKUP: return _NAME_LOOKUP[s]
    # 2. Prefix match (longest first)
    for pfx, sub in _PREFIX_RULES:
        if s.startswith(pfx) and len(s) > len(pfx) and s[len(pfx)].isalnum():
            return sub
    # 3. Keyword/substring match (longest phrase wins)
    for kw, sub in _KEYWORD_RULES:
        if kw in s:
            return sub
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    p.add_argument("--orth-csv",   type=Path, default=ORTH_CSV)
    p.add_argument("--syn-csv",    type=Path, default=SYN_CSV)
    p.add_argument("--fba-csv",    type=Path, default=FBA_CSV)
    p.add_argument("--out-long",   type=Path, default=OUT_LONG)
    p.add_argument("--out-feats",  type=Path, default=OUT_FEATS)
    p.add_argument("--out-summ",   type=Path, default=OUT_SUMM)
    p.add_argument("--out-uncl",   type=Path, default=OUT_UNCL)
    p.add_argument("--ess-rate-threshold", type=float, default=0.40,
                   help="branch is 'operationally essential' in an "
                        "organism if >= this fraction of its genes are "
                        "essential. Default 0.40.")
    args = p.parse_args()

    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    print("Loading inputs ...")
    labels = pd.read_csv(args.labels_csv)
    orth   = pd.read_csv(args.orth_csv) if args.orth_csv.exists() else None
    syn    = pd.read_csv(args.syn_csv)  if args.syn_csv.exists()  else None
    fba    = pd.read_csv(args.fba_csv)  if args.fba_csv.exists()  else None
    print(f"  labels: {len(labels):>7d} rows, {labels.organism.nunique()} organisms")

    # ---- diagnostic: what does gene_name look like across orgs? ----
    print(f"\n=== diagnostic: gene_name distribution ===")
    n_null = int(labels.gene_name.isna().sum())
    print(f"  null gene_name:    {n_null:>7d} ({100*n_null/len(labels):.1f}%)")
    is_beril = labels.organism.str.startswith("beril_").fillna(False)
    beril_lbl    = labels[is_beril]
    nonberil_lbl = labels[~is_beril]
    print(f"  BERIL rows:        {len(beril_lbl):>7d}  "
          f"({beril_lbl.gene_name.isna().sum()} null gene_name)")
    print(f"  non-BERIL rows:    {len(nonberil_lbl):>7d}  "
          f"({nonberil_lbl.gene_name.isna().sum()} null gene_name)")
    print(f"\n  top 15 BERIL gene_name values (non-null):")
    for n, c in beril_lbl.gene_name.dropna().value_counts().head(15).items():
        print(f"    {str(n)[:60]:<60s}  x{c}")
    print(f"\n  top 15 non-BERIL gene_name values (non-null):")
    for n, c in nonberil_lbl.gene_name.dropna().value_counts().head(15).items():
        print(f"    {str(n)[:60]:<60s}  x{c}")

    # ---- Pass 1: classify by gene_name ----
    print("\nPass 1: classify by gene_name (3 stages: exact / prefix / keyword) ...")
    labels["sub_branch"] = labels["gene_name"].apply(classify_by_name)
    n_named = int(labels.sub_branch.notna().sum())
    print(f"  classified by name:   {n_named:>7d} ({100*n_named/len(labels):.1f}%)")
    # Per-stage breakdown: how many classified each way?
    # Re-classify with each stage isolated, just for the diagnostic.
    def _exact_only(g):
        if g is None or pd.isna(g): return None
        s = str(g).strip().lower()
        return _NAME_LOOKUP.get(s)
    def _prefix_only(g):
        if g is None or pd.isna(g): return None
        s = str(g).strip().lower()
        for pfx, sub in _PREFIX_RULES:
            if s.startswith(pfx) and len(s) > len(pfx) and s[len(pfx)].isalnum():
                return sub
        return None
    sample = labels.gene_name.dropna()
    n_exact   = int(sum(1 for g in sample if _exact_only(g) is not None))
    n_prefix  = int(sum(1 for g in sample if _exact_only(g) is None
                                            and _prefix_only(g) is not None))
    n_keyword = n_named - n_exact - n_prefix
    print(f"    exact-name matches:  {n_exact:>7d}")
    print(f"    prefix matches:      {n_prefix:>7d}")
    print(f"    keyword/substring:   {n_keyword:>7d}")

    # ---- Pass 2: OG-vote propagation ----
    propagated = 0
    if orth is not None and "og_id" in orth.columns:
        print("\nPass 2: propagate via OG-vote ...")
        labels = labels.merge(orth[["organism","locus_tag","og_id"]],
                                on=["organism","locus_tag"], how="left")
        # vote per OG (majority of name-classified members)
        named = labels[labels.sub_branch.notna() & labels.og_id.notna()]
        og_votes: dict[str, str] = {}
        for og, grp in named.groupby("og_id"):
            top = grp.sub_branch.value_counts().idxmax()
            og_votes[og] = top
        unclassified_mask = labels.sub_branch.isna() & labels.og_id.notna()
        labels.loc[unclassified_mask, "sub_branch"] = \
            labels.loc[unclassified_mask, "og_id"].map(og_votes)
        propagated = int((labels.sub_branch.notna().sum()) - n_named)
        print(f"  propagated via OG:   +{propagated:>7d}  "
              f"(total classified: {labels.sub_branch.notna().sum()})")

    labels["branch"] = labels.sub_branch.map(_BRANCH_OF_SUB)
    n_class = int(labels.sub_branch.notna().sum())
    print(f"  TOTAL classified:    {n_class:>7d} ({100*n_class/len(labels):.1f}%)")
    print(f"  unclassified:        {len(labels)-n_class:>7d}")

    # ---- Branch coverage per organism ----
    print(f"\n=== Per-organism classification rate ===")
    cov = labels.groupby("organism").agg(
        n=("locus_tag","count"),
        n_class=("sub_branch", lambda s: s.notna().sum()),
    )
    cov["pct_class"] = 100 * cov.n_class / cov.n
    cov = cov.sort_values("pct_class")
    print(f"{'organism':<32s} {'n_genes':>8s} {'classified':>10s} {'pct':>6s}")
    for org, r in cov.iterrows():
        print(f"  {org:<30s} {int(r.n):>8d} {int(r.n_class):>10d} {r.pct_class:>5.1f}%")

    # ---- Long-format per (organism, sub_branch) ----
    classified = labels[labels.sub_branch.notna()].copy()
    # join feature aggregates (using last-fold's family_frac for the
    # per-row "typical" value; LOO-aware uses inside model)
    if orth is not None:
        ff_cols = [c for c in orth.columns if c.startswith("family_frac_essential_fold")]
        if ff_cols:
            orth["family_frac_mean"] = orth[ff_cols].mean(axis=1)
            classified = classified.merge(
                orth[["organism","locus_tag","family_frac_mean"]],
                on=["organism","locus_tag"], how="left")
    if syn is not None:
        classified = classified.merge(
            syn[["organism","locus_tag","max_synteny_count"]],
            on=["organism","locus_tag"], how="left")
    if fba is not None:
        classified = classified.merge(
            fba[["organism","locus_tag","fba_essential"]],
            on=["organism","locus_tag"], how="left")

    # Aggregate per (organism, sub_branch)
    grp_cols = ["organism","branch","sub_branch"]
    agg_dict: dict = {
        "n_genes":             ("locus_tag", "count"),
        "n_essential":         ("essential", "sum"),
    }
    if "family_frac_mean" in classified.columns:
        agg_dict["mean_family_frac"]   = ("family_frac_mean", "mean")
    if "max_synteny_count" in classified.columns:
        agg_dict["mean_max_synteny"]   = ("max_synteny_count", "mean")
    if "fba_essential" in classified.columns:
        agg_dict["fba_ess_rate"]       = ("fba_essential", "mean")
    long = classified.groupby(grp_cols, dropna=False).agg(**agg_dict).reset_index()
    long["ess_rate"] = long.n_essential / long.n_genes
    long = long.sort_values(["organism","branch","sub_branch"])
    long.to_csv(args.out_long, index=False)
    print(f"\nwrote {len(long)} rows to {args.out_long}")

    # ---- Cross-organism BRANCH-LEVEL derived signals ----
    # (a) function_essentiality_universal:
    #     For each sub_branch, fraction of organisms where ess_rate >= threshold.
    # (b) solution_diversity:
    #     For each sub_branch, count of distinct OG ids across all orgs.
    print(f"\n=== Branch-level derived features ===")
    print(f"  threshold for 'operationally essential': "
          f"ess_rate >= {args.ess_rate_threshold}")
    n_orgs = labels.organism.nunique()
    branch_summary = []
    for sub_id in sorted(_BRANCH_OF_SUB.keys()):
        sub_rows = long[long.sub_branch == sub_id]
        n_orgs_with_branch = int((sub_rows.n_genes > 0).sum())
        n_orgs_branch_ess  = int((sub_rows.ess_rate >= args.ess_rate_threshold).sum())
        func_ess = n_orgs_branch_ess / max(n_orgs, 1)
        if orth is not None and "og_id" in orth.columns:
            sub_genes = classified[classified.sub_branch == sub_id]
            n_distinct_og = int(sub_genes.og_id.nunique())
        else:
            n_distinct_og = -1
        branch_summary.append({
            "sub_branch":                       sub_id,
            "branch":                           _BRANCH_OF_SUB[sub_id],
            "n_organisms_with_branch":          n_orgs_with_branch,
            "n_organisms_branch_essential":     n_orgs_branch_ess,
            "function_essentiality_universal":  func_ess,
            "solution_diversity":               n_distinct_og,
        })

    print(f"\n{'sub_branch':<35s} {'orgs_pres':>9s} {'orgs_ess':>8s} "
          f"{'func_ess':>8s} {'sol_div':>7s}")
    for s in sorted(branch_summary, key=lambda r: -r["function_essentiality_universal"]):
        print(f"  {s['sub_branch']:<33s} {s['n_organisms_with_branch']:>9d} "
              f"{s['n_organisms_branch_essential']:>8d} "
              f"{s['function_essentiality_universal']:>7.2f}  "
              f"{s['solution_diversity']:>7d}")

    # ---- Emit per-gene blo_features.csv ----
    feat = classified[["organism","locus_tag","branch","sub_branch"]].copy()
    func_ess_map = {b["sub_branch"]: b["function_essentiality_universal"]
                     for b in branch_summary}
    sol_div_map  = {b["sub_branch"]: b["solution_diversity"]
                     for b in branch_summary}
    feat["function_essentiality_universal"] = feat["sub_branch"].map(func_ess_map)
    feat["solution_diversity"]              = feat["sub_branch"].map(sol_div_map)
    # log-scale solution diversity (1 -> 0, 10 -> 1, 100 -> 2 ...)
    import math
    feat["log_solution_diversity"] = feat["solution_diversity"].apply(
        lambda v: math.log10(max(v, 1)) if v >= 0 else float("nan"))
    feat.to_csv(args.out_feats, index=False)
    print(f"\nwrote {len(feat)} rows to {args.out_feats}")
    print(f"  feature distributions:")
    print(f"    function_essentiality_universal: "
          f"min={feat['function_essentiality_universal'].min():.2f} "
          f"med={feat['function_essentiality_universal'].median():.2f} "
          f"max={feat['function_essentiality_universal'].max():.2f}")
    print(f"    log_solution_diversity:          "
          f"min={feat['log_solution_diversity'].min():.2f} "
          f"med={feat['log_solution_diversity'].median():.2f} "
          f"max={feat['log_solution_diversity'].max():.2f}")

    # ---- Save unclassified for review ----
    uncl = labels[labels.sub_branch.isna()].copy()
    uncl_by_name = uncl.gene_name.fillna("(none)").value_counts().head(200)
    uncl_top = pd.DataFrame({
        "gene_name": uncl_by_name.index,
        "n_orgs_with_this_name": uncl_by_name.values,
    })
    uncl_top.to_csv(args.out_uncl, index=False)
    print(f"\nwrote top-200 unclassified gene names to {args.out_uncl}")
    print(f"  top 10 unclassified gene names (review these to extend rules):")
    for n, c in uncl_by_name.head(10).items():
        print(f"    {n:<30s} appears in {c:>5d} (organism, locus) rows")

    # ---- Summary JSON ----
    summary = {
        "n_total_rows": len(labels),
        "n_classified": int(labels.sub_branch.notna().sum()),
        "n_unclassified": int(labels.sub_branch.isna().sum()),
        "pct_classified": 100 * float(labels.sub_branch.notna().sum()) / len(labels),
        "by_name": n_named,
        "by_og_vote": propagated,
        "ess_rate_threshold": args.ess_rate_threshold,
        "branches": branch_summary,
        "top_unclassified_names": [
            {"gene_name": str(n), "count": int(c)}
            for n, c in uncl_by_name.head(50).items()
        ],
    }
    with open(args.out_summ, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {args.out_summ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
