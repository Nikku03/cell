# Streaming a 750 MB slice out of a 21 GB gzipped GCTX without downloading it

LINCS L1000 Level 5 (GSE92742) is 21.3 GB compressed, 23.4 GB decompressed. This session had
8.7 GB of free disk. The slice actually needed -- shRNA knockdown signatures at the 978 measured
landmark genes -- is 750 MB.

The obstacle is that gzip is not seekable, so the loop-224 trick (HTTP range requests straight into
a contiguous HDF5 dataset) does not apply directly. Two facts make it work anyway:

1. HDF5 wrote this file's METADATA AT THE END. `h5py` on a truncated prefix fails with
   "bad symbol table node signature" because the root group's symbol table is past the prefix.
   The fix is a SPARSE file: `truncate -s 23380287022` gives h5py the file size it demands while
   costing no disk, and the head and tail can then be written into their true offsets.
2. The matrix is CONTIGUOUS and stored one signature per row:
       /0/DATA/0/matrix  (473647, 12328) float32, layout=contiguous, offset=6240
   so signature i occupies bytes 6240 + i*49312 .. +49312, and a single sequential pass can pick
   out any subset of rows.

`stream_metadata.py` -- pass 1. Sequential byte-ranges of the .gz fed into ONE continuous
zlib decompressobj, so a dropped connection resumes at the compressed offset instead of restarting.
Keeps only the first 64 MB and last 320 MB, written into the sparse file at their true offsets.
Result: full 23,380,287,022 bytes verified, 429 MB on disk.

`stream_extract.py` -- pass 2. Same resumable stream; for each wanted row, accumulates its 49,312
bytes across chunk boundaries, subsets to the 978 landmark columns, writes to a memmap.
Result: 191,713 of 191,713 rows in 532 s.

Verified by positive control: knocking down gene g gives g's OWN landmark measurement a mean
z-score of -2.2997 (median -2.07) against -0.0058 for random other landmarks in the same rows;
97.1% of the 946 knocked-down genes that are themselves landmarks have own-gene z < 0.
