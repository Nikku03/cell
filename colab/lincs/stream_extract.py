import urllib.request, zlib, numpy as np, time
U=("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
   "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz")
CSZ=21328033748; BASE=6240; NG=12328; RB=NG*4
S=np.load("select.npz", allow_pickle=True)
rows=S["rows"]; lm=S["lmcols"]
out=np.lib.format.open_memmap("shrna_landmark.npy", mode="w+",
                              dtype=np.float32, shape=(len(rows), len(lm)))
print(f"extracting {len(rows):,} rows x {len(lm)} landmarks", flush=True)
dec=zlib.decompressobj(16+zlib.MAX_WBITS)
CHUNK=64*1024*1024
cpos=0; pos=0; k=0; fails=0; got=0; t0=time.time()
partial=bytearray()
while cpos < CSZ and k < len(rows):
    hi=min(cpos+CHUNK, CSZ)-1
    try:
        req=urllib.request.Request(U, headers={"Range":f"bytes={cpos}-{hi}"})
        blob=urllib.request.urlopen(req, timeout=300).read()
    except Exception as e:
        fails+=1
        if fails>60: print("TOO MANY FAILURES", flush=True); break
        time.sleep(min(2**min(fails,5),30)); continue
    cpos+=len(blob)
    o=dec.decompress(blob)
    if not o: continue
    end=pos+len(o)
    while k < len(rows):
        rs=BASE+int(rows[k])*RB; re=rs+RB
        if re<=pos: k+=1; partial=bytearray(); continue
        if rs>=end: break
        a=max(rs,pos); b=min(re,end)
        partial+=o[a-pos:b-pos]
        if b==re:
            out[k]=np.frombuffer(bytes(partial),dtype="<f4")[lm]
            partial=bytearray(); k+=1; got+=1
        else: break
    pos=end
    if cpos % (2*1024*1024*1024) < CHUNK:
        el=time.time()-t0
        print(f"  {cpos/1e9:5.1f}/{CSZ/1e9:.1f} GB compressed, {got:,}/{len(rows):,} rows "
              f"({cpos/1e6/max(el,1):.0f} MB/s, {el:.0f}s, {fails} retries)", flush=True)
out.flush()
print(f"DONE extracted {got:,}/{len(rows):,} rows, retries={fails}, {time.time()-t0:.0f}s", flush=True)
