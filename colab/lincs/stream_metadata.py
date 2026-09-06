import urllib.request, zlib, os, time
U=("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
   "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz")
CSZ=21328033748; EOFSZ=23380287022
HEAD=64*1024*1024; TAIL=EOFSZ-320*1024*1024
CHUNK=64*1024*1024
f=os.open("prefix.h5", os.O_RDWR)
dec=zlib.decompressobj(16+zlib.MAX_WBITS)
cpos=0; pos=0; kept=0; t0=time.time(); fails=0
while cpos < CSZ:
    hi=min(cpos+CHUNK, CSZ)-1
    try:
        req=urllib.request.Request(U, headers={"Range":f"bytes={cpos}-{hi}"})
        blob=urllib.request.urlopen(req, timeout=300).read()
    except Exception as e:
        fails+=1
        if fails>40: print("TOO MANY FAILURES", flush=True); break
        time.sleep(min(2**min(fails,5),30)); continue
    cpos += len(blob)
    out=dec.decompress(blob)
    if out:
        end=pos+len(out)
        if pos<HEAD or end>TAIL: os.pwrite(f, out, pos); kept+=len(out)
        pos=end
    if cpos % (2*1024*1024*1024) < CHUNK:
        el=time.time()-t0
        print(f"  compressed {cpos/1e9:5.1f}/{CSZ/1e9:.1f} GB -> {pos/1e9:5.1f} GB out "
              f"({cpos/1e6/max(el,1):.0f} MB/s, {el:.0f}s), kept {kept/1e6:.0f} MB, {fails} retries", flush=True)
os.close(f)
print(f"DONE out={pos:,} expected={EOFSZ:,} kept={kept/1e6:.0f} MB retries={fails} in {time.time()-t0:.0f}s", flush=True)
