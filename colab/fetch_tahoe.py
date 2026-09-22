"""fetch_tahoe — download the RIGHT Tahoe-100M table straight from HuggingFace (faster + complete vs the Drive
copy). Grabs ONLY cell_eval/ (14 wide pseudobulk plates, ~900 MB) — never pdex (the 4.1B-row long table) and
never the per-cell embeddings.

Dataset: tahoebio/tahoe-de-rhaister (CC0, public). Each plate is ~4,400 conditions x ~2,000 genes; ~1,800 of
those genes match our index. signal_combiner builds per-gene drug-response vectors from these and CACHES them
(outputs/orphan/tahoe_vecs.npz), which persist.py saves to Drive — so this 900 MB download happens ONCE.
"""
import os, sys, urllib.request

REPO = "tahoebio/tahoe-de-rhaister"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/cell_eval/"
PLATES = [f"plate_plate{i}.parquet" for i in range(1, 15)]


def fetch(dst="tahoe_hf/cell_eval", retries=3):
    """download the 14 cell_eval plates (skipping any already present) -> returns the parent dir for
    TAHOE_DE_DIR, or None if nothing landed."""
    # if the derived cache already exists (restored from Drive), no download is needed at all
    if os.path.exists("outputs/orphan/tahoe_vecs.npz"):
        print("tahoe: derived vectors already cached (outputs/orphan/tahoe_vecs.npz) — skipping download")
        return None
    os.makedirs(dst, exist_ok=True)
    got = 0
    for f in PLATES:
        out = os.path.join(dst, f)
        if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
            got += 1; continue
        for attempt in range(retries):
            try:
                urllib.request.urlretrieve(BASE + f, out)
                got += 1
                print(f"  {f}  ({os.path.getsize(out)/1e6:.0f} MB)")
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  skip {f}: {str(e)[:60]}")
    print(f"tahoe cell_eval: {got}/14 plates in {dst}")
    return os.path.dirname(os.path.abspath(dst)) if got else None


if __name__ == "__main__":
    d = fetch(sys.argv[1] if len(sys.argv) > 1 else "tahoe_hf/cell_eval")
    print("TAHOE_DE_DIR=", d)
