"""fetch_replogle -- reproducible fetch+parse of the external Replogle K562-essential Perturb-seq (for transformer_ko6 / v6).

Source: HuggingFace dataset nicolas-lynn/replogle-perturb, k562ess (CRISPRi KNOCKDOWN, z-units -- same readout/direction as our K562 data).
Downloads perturbation_zscores.parquet (1,971 perturbations x 8,563 genes), keeps the 630 knockouts NOT already in our nlz_K562.pkl, stores
each as a sparse {gene: z} dict (genes with |z|>=0.5, capped 400/KO) matching our nlz format -> nlz_Replogle.pkl.

(Norman was also on HF but is CRISPR-ACTIVATION -- wrong direction for a knockout model -- so it is intentionally NOT used.)
"""
import os
import pickle
from pathlib import Path
import numpy as np
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
URL = "https://huggingface.co/datasets/nicolas-lynn/replogle-perturb/resolve/main/k562ess/perturbation_zscores.parquet"


def main():
    import subprocess, pyarrow.parquet as pq
    pq_path = SP / "repl_k562_zscores.parquet"
    if not pq_path.exists():
        subprocess.run(["curl", "-sSL", URL, "-o", str(pq_path)], check=True)
    df = pq.read_table(str(pq_path)).to_pandas()
    perts = list(df.index); genes = [c for c in df.columns if c != "__index_level_0__"]
    ours = set(pickle.load(open(SP / "nlz_K562.pkl", "rb")))
    new_kos = [p for p in perts if p not in ours]
    M = df[genes].values.astype(np.float32); pidx = {p: i for i, p in enumerate(perts)}
    extra = {}
    for ko in new_kos:
        row = M[pidx[ko]]; keep = np.where(np.abs(row) >= 0.5)[0]
        if len(keep) > 400:
            keep = keep[np.argsort(-np.abs(row[keep]))[:400]]
        extra[ko] = {genes[j]: float(row[j]) for j in keep}
    pickle.dump(extra, open(SP / "nlz_Replogle.pkl", "wb"))
    print(f"nlz_Replogle.pkl: {len(extra)} new K562 KOs (of {len(perts)} Replogle perturbations; {len(ours)} already ours)")


if __name__ == "__main__":
    main()
