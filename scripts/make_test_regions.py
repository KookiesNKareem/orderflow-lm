"""Create disjoint evaluation regions within the held-out test split."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oflm.sim import find_regions, replay  # noqa: E402
from oflm import tokenizer as tk  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="data/tokens_v2")
    args = p.parse_args()
    tok = Path(args.tokens)

    val = np.fromfile(tok / "val.bin", dtype=np.uint16)
    test = np.fromfile(tok / "test.bin", dtype=np.uint16)
    assert len(val) == len(test), (
        f"val/test length mismatch ({len(val)} vs {len(test)}): the equal-half "
        "assumption fails; rebuild splits from normalized data instead")

    regions = find_regions(val)
    assert regions and regions[0][1] == 0, "val.bin does not start with a day header"
    lengths = [e - s for _, s, e in regions]
    assert sum(lengths) == len(test), "val regions do not tile the file"

    out, meta = [], []
    pos = 0
    for (sid, _, _), L in zip(regions, lengths):
        chunk = test[pos:pos + L]
        pos += L
        dt = np.flatnonzero((chunk >= tk.DT0) & (chunk < tk.DT0 + 32))
        assert len(dt), f"no Delta-t token found in chunk for symbol id {sid}"
        resync = int(dt[0])
        out.append(np.array([tk.BOS, tk.SYM0 + sid], dtype=np.uint16))
        out.append(chunk[resync:])
        meta.append({"sym_id": sid, "chunk_tokens": int(L), "resync_skip": resync})
        print(f"sym {sid}: chunk {L:,} tokens, resynced past {resync} tokens")

    fixed = np.concatenate(out)
    fixed.tofile(tok / "test_fixed.bin")
    with open(tok / "test_fixed.bin.meta.json", "w") as f:
        json.dump({"vocab": tk.VOCAB, "dtype": "uint16", "tokens": int(len(fixed)),
                   "source": "test.bin repaired by make_test_regions.py",
                   "chunks": meta}, f)

    # validation: regions found, and a replay smoke test per region head
    new_regions = find_regions(fixed)
    assert len(new_regions) == len(regions), "region count mismatch after repair"
    for sid, s, e in new_regions:
        head = replay(fixed[s:min(s + 200_000, e)])
        n_tr = len(head["trade_idx"])
        print(f"sym {sid}: region [{s:,}, {e:,}), smoke replay {n_tr:,} trades ok")
    print(f"wrote {tok / 'test_fixed.bin'} ({len(fixed):,} tokens)")


if __name__ == "__main__":
    main()
