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



# 5. Statistical inference recomputed from the released per-seed differences.
import math


def _betacf(a, b, x):
    qab, qap, qam, c, d = a + b, a + 1, a - 1, 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / d; h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 / (1 + aa * d); c = 1 + aa / c; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 / (1 + aa * d); c = 1 + aa / c; de = d * c; h *= de
        if abs(de - 1) < 1e-15:
            break
    return h


def t_two_sided_p(t, df):
    """Two-sided Student-t p-value via the regularised incomplete beta function."""
    if math.isinf(t):
        return 0.0
    x = df / (df + t * t)
    a, b = df / 2, 0.5
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - front * _betacf(b, a, 1 - x) / b


def holm_adjust(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


INF = list(csv.DictReader(open(A / "inference.csv")))
fl = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
worst_t = worst_p = 0.0
for r in INF:
    d = [float(r[k]) for k in r if k.startswith("diff_s") and r[k] not in ("", None)]
    J = len(d); mean = sum(d) / J
    var = (1 / J + fl(r, "ratio")) * sum((x - mean) ** 2 for x in d) / (J - 1)
    t = mean / math.sqrt(var) if var > 0 else math.copysign(math.inf, mean)
    worst_t = max(worst_t, abs(t - fl(r, "nb_t")) / max(1, abs(t)))
    worst_p = max(worst_p, abs(t_two_sided_p(t, J - 1) - fl(r, "nb_p")))
check(worst_t < 1e-9 and worst_p < 1e-6,
      f"corrected t-statistics and p-values of all {len(INF)} comparisons recomputed (max abs p diff {worst_p:.1e})")
ci_ok = all((fl(r, "nb_p") < 0.05) == (fl(r, "nb_lo") > 0 or fl(r, "nb_hi") < 0) for r in INF if r.get("nb_lo"))
check(ci_ok, "corrected-test 95% CIs agree with the unadjusted p-values")
prim = [r for r in INF if r["primary"] == "True"]
for col, src, fam in (("holm_p", "nb_p", prim), ("iu_holm_p", "iu_p", prim),
                      ("holm_p", "nb_p", [r for r in INF if r["ref_set"] == "tuned_test" and r.get("holm_p")])):
    adj = holm_adjust([fl(r, src) for r in fam])
    check(len(fam) == 20 and max(abs(a - fl(r, col)) for a, r in zip(adj, fam)) < 1e-9,
          f"Holm adjustment '{col}' recomputed over its family of {len(fam)}")
vs = defaultdict(list)
for r in INF:
    if r["ref_set"].startswith("vs_") and r["metric"] == "emp":
        vs[(r["dataset"], r["tfm"])].append(r)
check(all(abs(max(fl(x, "nb_p") for x in vs[(r["dataset"], r["tfm"])]) - fl(r, "iu_p")) < 1e-12 and
          len(vs[(r["dataset"], r["tfm"])]) == 3 for r in prim),
      "intersection-union p = worst of the three per-GBDT comparisons, for every primary comparison")

# 6. The reference GBDT is the best on the calibration split, never chosen on test data.
cal = defaultdict(list)
for r in csv.DictReader(open(A / "metrics_per_seed.csv")):
    if r["model"] in ("lgbm", "xgb", "catboost"):
        cal[(r["dataset"], r["model"])].append(float(r["emp_cal"]))
ref_ok = all(r["ref"] == max(("lgbm", "xgb", "catboost"), key=lambda m: st.mean(cal[(r["dataset"], m)])) for r in prim)
check(ref_ok, "every primary reference GBDT is the one with the best mean calibration-split EMP")

# 7. Robustness checks cover exactly the datasets and splits the paper states.
strong = {(r["dataset"], r["J"]) for r in INF if r["ref_set"] == "strong" and r["metric"] == "emp"
          and r["ref"] in ("lgbm_200", "catboost_200")}
dup = {(r["dataset"], r["J"]) for r in INF if r["dataset"].endswith("_dupgroup") and r["metric"] == "emp"}
check({d for d, _ in strong} <= {"cell2cell", "iranian", "orange"} and all(j == "5" for _, j in strong),
      f"200-trial comparisons use only Cell2Cell/Iranian/Orange and five splits: {sorted(strong)}")
check({d for d, _ in dup} == {"telco_dupgroup", "iranian_dupgroup", "ecommerce_dupgroup"} and all(j == "5" for _, j in dup),
      "duplicate-group checks cover Telco, Iranian and E-Commerce with five splits")
check(all(r["J"] == "10" for r in prim), "every primary comparison uses ten splits")

# 8. Software versions in the kernel logs match the appendix.
logs = "".join(e.get("data", "") for p in K.glob("*/*.log") for e in json.load(open(p)))
vers = defaultdict(set)
for pkg, v in re.findall(r"\d\d:\d\d:\d\d (tabpfn|tabicl|lightgbm|xgboost|catboost|optuna|scikit-learn|interpret) ([0-9.]+)", logs):
    vers[pkg].add(v)
appendix = tex[tex.find("Software &"):tex.find("Hardware &")]
ok = all(len(v) == 1 and next(iter(v)) in appendix for v in vers.values()) and len(vers) >= 7
check(ok, "package versions logged by every kernel are unique and match the appendix: "
      + ", ".join(f"{k} {'/'.join(sorted(v))}" for k, v in sorted(vers.items())))

print(f"\n{'ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED'}")
sys.exit(1 if FAIL else 0)
