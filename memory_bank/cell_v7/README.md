# cell_v7: what clinical trials and drug development revealed about each gene

Written by `rem/atlas/cellbank7.py build`. This is a layer on top of `cell_v6`: the records are unchanged, so read them from `cell_v6`.

## The idea

When a drug that blocks a gene is given to people, the trials and post-marketing reports record what blocking that gene does in a whole human body. That is evidence about the gene's job, and the cell model had not used it.

Only drugs whose action on the gene is loss of function count (inhibitor, antagonist, blocker and similar). A knockdown is also loss of function, so the two are comparable.

## Files

| File | Contents |
|---|---|
| `clinical_profiles.json` | A 67-number profile per gene (1,882 genes), with `layout` naming each number |
| `clinical_facts.json` | The same knowledge as readable text per gene (see below) |
| `build_report.json` | Coverage counts |

### What a profile holds

- **Patient adverse events** (openFDA significant reactions), grouped into 14 organ or process categories by fixed keyword rules: blood, liver, heart, kidney, gut, skin, nerve, infection, metabolic, lung, immune, bleeding-vascular, tumour, death. For each category: the share of the gene's drugs with such a reaction, and the mean log1p summed log-likelihood ratio.
- **Drug warnings:** the 12 most common toxicity classes.
- **Trials:** the number of trials, the share stopped, the share stopped for each reason (negative result, safety, business, enrolment, design, other), and the highest clinical stage.
- **Indications:** the share of the gene's drugs developed for blood cancers, and for any cancer.
- **Preclinical safety events** on the gene.

### What a readable fact holds

- the drugs that block the gene;
- the highest clinical stage;
- what happened in patients, with example events;
- trials run, trials stopped and why;
- warnings;
- whether the gene's drugs were developed for blood cancer;
- preclinical safety findings.

## Source

Open Targets Platform 26.09 (CC0). It draws on ChEMBL, ClinicalTrials.gov, DailyMed, openFDA, ToxCast and the literature.

## Caveats

- The keyword categories are coarse.
- In patients, a drug's on-target effects, off-target effects and the disease itself are mixed together. For example, openFDA "tumour" events often record the indication rather than a side effect.
- Only genes that are drug targets have a profile.
