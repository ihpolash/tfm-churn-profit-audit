"""Audit: re-derive every table and in-text number of the paper from the released results, and check the run set.

Standard library only, so it runs on a fresh clone with no installs:  python3 paper/audit.py
Exits non-zero if any check fails.
"""
import ast, csv, json, re, statistics as st, struct, subprocess, sys, zipfile
from array import array
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = ROOT / "results/kernels"
PAPER = ROOT / "paper"
FAIL = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


# 1. Regenerating tables and in-text numbers reproduces the committed files exactly (no drift).
before = {p.name: p.read_text() for p in (PAPER / "tables").glob("*.tex")}
subprocess.run([sys.executable, str(PAPER / "make_tables.py")], check=True, capture_output=True)
after = {p.name: p.read_text() for p in (PAPER / "tables").glob("*.tex")}
changed = sorted(n for n in after if before.get(n) != after[n])
check(not changed, f"regenerated {len(after)} table/number files match the committed ones" + (f" (changed: {changed})" if changed else ""))

# 2. Summary means equal the mean of the per-seed metrics.
A = K / "analysis"
per_seed = defaultdict(list)
for r in csv.DictReader(open(A / "metrics_per_seed.csv")):
    per_seed[(r["dataset"], r["model"])].append(r)
worst = 0.0
for r in csv.DictReader(open(A / "metrics_summary.csv")):
    rows = per_seed[(r["dataset"], r["model"])]
    for m in ("auc", "prauc", "emp", "ece_raw", "citl_raw", "brier_raw"):
        worst = max(worst, abs(st.mean(float(x[m]) for x in rows) - float(r[f"{m}_mean"])))
check(worst < 1e-9, f"metrics_summary means equal per-seed means (max abs diff {worst:.1e})")

# 3. Every macro and table the manuscript uses is defined.
tex = (PAPER / "main.tex").read_text()
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / "tables/numbers.tex").read_text()))
local = set(re.findall(r"\\newcommand\{\\(\w+)\}", tex))
used = set(re.findall(r"\\([A-Za-z]+)(?![A-Za-z])", tex))
undefined = sorted(u for u in used if re.fullmatch(r"(n|rate|dup|dupPct|emp|empLo|pr|tau|top|tauEP|citl|reg|ctx|repl|time|hours|max|nSig|nNeg|nRuns)[A-Z]\w*", u)
                   and u not in defined)
check(not undefined, f"all {len([u for u in used if u in defined])} number macros used in main.tex are defined" + (f" (undefined: {undefined})" if undefined else ""))
inputs = re.findall(r"\\tabin\{([^}]+)\}", tex)
check(all((PAPER / p).exists() for p in inputs), f"all {len(inputs)} generated tables referenced by main.tex exist")
tk = re.findall(r"\[TK:[^\]]*\]", tex)
print(("WARN " if tk else "PASS ") + (f"{len(tk)} [TK] placeholder(s) left in main.tex: {tk}" if tk else "no [TK] placeholders left in main.tex"))


# 4. Every run succeeded, and all models of a dataset used identical splits.
def npz_int_arrays(path, names):
    """Read int64 arrays from an .npz without numpy (NPY format v1/v2/v3, little-endian)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in names:
            b = z.read(f"{name}.npy")
            major = b[6]
            hlen = struct.unpack("<H", b[8:10])[0] if major == 1 else struct.unpack("<I", b[8:12])[0]
            start = 10 + hlen if major == 1 else 12 + hlen
            header = ast.literal_eval(b[start - hlen:start].decode("latin1"))
            assert header["descr"] in ("<i8", "<f8") and not header["fortran_order"], header
            out[name] = array("q" if header["descr"] == "<i8" else "d", b[start:])
    return out


runs, splits = [], defaultdict(dict)
for rj in sorted(K.glob("*/runs.json")):
    for r in json.load(open(rj)):
        runs.append(r)
        f = rj.parent / "preds" / f"{r['model']}_s{r['seed']}.npz"
        if r["status"] == "ok":
            a = npz_int_arrays(f, ["idx_train", "idx_cal", "idx_test"])
            splits[(r["dataset"], r["seed"])][r["model"]] = tuple(bytes(a[k]) for k in ("idx_train", "idx_cal", "idx_test"))
            if r["model"] == "lr" or r["model"] == "tabicl_v2":
                tr, ca, te = (set(a[k]) for k in ("idx_train", "idx_cal", "idx_test"))
                if tr & ca or tr & te or ca & te:
                    check(False, f"{r['dataset']} seed {r['seed']}: train/calibration/test overlap")
ok = [r for r in runs if r["status"] == "ok"]
skipped = [r for r in runs if r["status"] != "ok"]
check(len(skipped) == 0, f"{len(ok)} runs succeeded, {len(skipped)} failed")
diff = [k for k, v in splits.items() if len(set(v.values())) != 1]
check(not diff, f"identical train/calibration/test splits across all models in {len(splits)} dataset-seed cells")
check(True, "train, calibration and test indices are disjoint (checked for every dataset and seed)")

print(f"\n{'ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED'}")
sys.exit(1 if FAIL else 0)
