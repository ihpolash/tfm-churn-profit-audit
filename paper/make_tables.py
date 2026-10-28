"""Generate every results table and in-text number of the paper from the stored analysis outputs.

Reads the released kernel outputs in results/kernels/; writes paper/tables/*.tex and
paper/tables/numbers.tex (LaTeX macros used in the prose). Standard library only. Rerun after any re-analysis.
"""
import csv, json, os, re, shutil, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = ROOT / "results/kernels"
A = Path(os.environ.get("CHURN_ANALYSIS", K / "analysis"))  # override only for development
TAB = Path(__file__).resolve().parent / "tables"
TAB.mkdir(exist_ok=True)

MAIN = ["telco", "maven", "cell2cell", "iranian", "orange"]
DS_NAME = {"telco": "Telco", "maven": "Maven", "cell2cell": "Cell2Cell", "iranian": "Iranian", "orange": "Orange",
           "ecommerce": "E-Commerce"}
DS_KEY = {"telco": "Telco", "maven": "Maven", "cell2cell": "CellTwoCell", "iranian": "Iranian", "orange": "Orange",
          "ecommerce": "ECommerce"}  # letters only, for LaTeX macro names
MODELS = ["lr", "ebm", "lgbm", "xgb", "catboost", "tabicl_v2", "tabpfn_2_5", "tabpfn_3_5", "tabpfn_3_5_fast"]
M_NAME = {"lr": "Logistic regression", "ebm": "EBM", "lgbm": "LightGBM", "xgb": "XGBoost", "catboost": "CatBoost",
          "tabicl_v2": "TabICL v2", "tabpfn_2_5": "TabPFN-2.5", "tabpfn_3_5": "TabPFN-3.5",
          "tabpfn_3_5_fast": "TabPFN-3.5-Fast", "xgb_default": "XGBoost (library default)",
          "lgbm_200": "LightGBM (200 trials)", "catboost_200": "CatBoost (200 trials)"}
TFMS = ["tabicl_v2", "tabpfn_2_5", "tabpfn_3_5", "tabpfn_3_5_fast"]
MACROS = {}


def rows(name):
    return list(csv.DictReader(open(A / name)))


def f(x, d=3):
    return f"{float(x):.{d}f}"


def unsign_zero(s):
    """A value that rounds to zero is printed as 0.0, not +0.0 or -0.0."""
    return re.sub(r"(?<![\d.])[+-](0\.0+)(?![\d])", r"\1", s)


def macro(name, value):
    assert name.isalpha(), name
    MACROS[name] = unsign_zero(value)


def write(name, body):
    (TAB / name).write_text(unsign_zero(body))


summ = {(r["dataset"], r["model"]): r for r in rows("metrics_summary.csv")}
boot = {(r["dataset"], r["metric"], r["tfm"], r["ref_gbdt"]): r for r in rows("bootstrap_vs_best_gbdt.csv")}
best_ref = {}  # (dataset, metric) -> best tuned GBDT used as reference
for (ds, met, tfm, ref) in boot:
    if ref != "xgb_default":
        best_ref[(ds, met)] = ref
sig = lambda r: float(r["ci_lo"]) > 0 or float(r["ci_hi"]) < 0

# ---------------------------------------------------------------- Table 1: datasets
PROV = {"telco": ("IBM sample data (fictional)", "\\cite{telco}"), "maven": ("Maven Analytics (fictional)", "\\cite{maven}"),
        "cell2cell": ("US wireless carrier", "\\cite{cell2cell}"), "iranian": ("Iranian telecom operator", "\\cite{iranian}"),
        "orange": ("Orange, KDD Cup 2009", "\\cite{guyon2009kddcup}"), "ecommerce": ("Kaggle, origin unstated", "\\cite{ecommerce}")}
lines = []
for ds in MAIN + ["ecommerce"]:
    m = json.load(open(K / f"{ds}-cpu/data_meta.json"))
    ndup = m.get("n_dup_removed") or 0  # kernels run before de-duplication existed had no duplicates (timing log)
    DUPS = {ds: ndup}
    macro(f"dup{DS_KEY[ds]}", f"{ndup:,}")
    macro(f"dupPct{DS_KEY[ds]}", f"{100 * ndup / (m['n'] + ndup):.1f}")
    eprof = "yes" if ds in ("telco", "maven", "cell2cell") else "no"
    role = "replication" if ds == "ecommerce" else "main"
    lines.append(f"{DS_NAME[ds]} {PROV[ds][1]} & {PROV[ds][0]} & {m['n']:,} & {m['n_feat']} & {100 * m['churn_rate']:.1f} "
                 f"& {DUPS.get(ds, 0)} & {eprof} & {role} \\\\")
    macro(f"n{DS_KEY[ds]}", f"{m['n']:,}")
    macro(f"rate{DS_KEY[ds]}", f"{100 * m['churn_rate']:.1f}")
write("datasets.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table 2: EMP and PR-AUC per model and dataset
def best_of(ds, col):
    return max(MODELS, key=lambda m: float(summ[(ds, m)][f"{col}_mean"]))


lines = []
for m in MODELS:
    cells = []
    for ds in MAIN:
        for col, d in (("prauc", 3), ("emp", 3)):
            v = f(summ[(ds, m)][f"{col}_mean"], d)
            cells.append(f"\\textbf{{{v}}}" if best_of(ds, col) == m else v)
    if m == "tabicl_v2":
        lines.append("\\midrule")
    lines.append(f"{M_NAME[m]} & " + " & ".join(cells) + " \\\\")
write("main_results.tex", "\n".join(lines) + "\n")

lines = []
for m in MODELS:
    cells = []
    for ds in MAIN:
        v = f(summ[(ds, m)]["auc_mean"], 3)
        cells.append(f"\\textbf{{{v}}}" if best_of(ds, "auc") == m else v)
    if m == "tabicl_v2":
        lines.append("\\midrule")
    lines.append(f"{M_NAME[m]} & " + " & ".join(cells) + " \\\\")
write("auc_results.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table 3: inference (split-aware)
inf = {(r["dataset"], r["metric"], r["tfm"], r["ref_set"]): r for r in rows("inference.csv")}
num = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
hier_sig = lambda r: num(r, "hier_lo") > 0 or num(r, "hier_hi") < 0
holm_sig = lambda r: r.get("holm_p") not in (None, "") and num(r, "holm_p") < 0.05


def cell(ds, met, tfm, refset="tuned"):
    """Difference with markers: * Holm-adjusted p < 0.05 (primary family only); dagger: two-level interval excludes 0."""
    r = inf[(ds, met, tfm, refset)]
    txt = f"{num(r, 'rel_pct'):+.1f}\\%" if met == "emp" else f"{100 * num(r, 'mean_diff'):+.1f}"
    mark = "^{\\ast}" if holm_sig(r) else ("^{\\dagger}" if hier_sig(r) else "")
    return f"$\\mathbf{{{txt}}}{mark}$" if mark else f"${txt}$"


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


lines = []
for ds in MAIN:
    for i, tfm in enumerate(TFMS):
        r = inf[(ds, "emp", tfm, "tuned")]
        bold = (lambda x: f"\\textbf{{{x}}}") if holm_sig(r) else (lambda x: x)
        lines.append(("\\multirow{4}{*}{" + DS_NAME[ds] + "}" if i == 0 else "")
                     + f" & {M_NAME[tfm]} & ${num(r, 'mean_diff'):+.3f}$ & ${num(r, 'rel_pct'):+.1f}$ & "
                     f"$[{num(r, 'nb_lo'):+.3f}, {num(r, 'nb_hi'):+.3f}]$ & {r['seeds_positive']}/{r['J']} & "
                     f"{pfmt(num(r, 'nb_p'))} & {bold(pfmt(num(r, 'holm_p')))} & "
                     f"{(lambda x: f'\\textbf{{{x}}}' if num(r, 'iu_holm_p') < 0.05 else x)(pfmt(num(r, 'iu_holm_p')))} \\\\")
    if ds != MAIN[-1]:
        lines.append("\\midrule")
write("primary.tex", "\n".join(lines) + "\n")
prim = [r for r in inf.values() if r.get("primary") == "True"]
macro("nPrimary", str(len(prim)))
macro("nHolmSig", str(sum(holm_sig(r) for r in prim)))
macro("nIUSig", str(sum(num(r, "iu_holm_p") < 0.05 for r in prim)))
test_sel = [r for r in inf.values() if r["ref_set"] == "tuned_test" and r.get("holm_p") not in (None, "")]
macro("nHolmSigTest", str(sum(num(r, "holm_p") < 0.05 for r in test_sel)))
macro("nRefChanged", str(sum(inf[(r["dataset"], "emp", r["tfm"], "tuned_test")]["ref"] != r["ref"] for r in prim) // 4))
sig_cases = lambda col: sorted({f"{M_NAME[r['tfm']]} on {DS_NAME[r['dataset']]}" for r in prim if num(r, col) < 0.05})
macro("iuSigCases", "; ".join(sig_cases("iu_holm_p")) or "none")
macro("nSeeds", str(max(int(r["J"]) for r in prim)))
macro("minHolm", pfmt(min(num(r, "holm_p") for r in prim)))
best = min(prim, key=lambda r: num(r, "holm_p"))
macro("minHolmCase", f"{M_NAME[best['tfm']]} on {DS_NAME[best['dataset']]}")
for t, k in (("tabpfn_3_5", "PFNThree"), ("tabicl_v2", "ICL"), ("tabpfn_2_5", "PFNTwo"), ("tabpfn_3_5_fast", "PFNFast")):
    rs = [inf[(ds, "emp", t, "tuned")] for ds in MAIN]
    macro(f"nHierEMP{k}", str(sum(hier_sig(r) and num(r, "mean_diff") > 0 for r in rs)))
    macro(f"nNegEMP{k}", str(sum(num(r, "hier_hi") < 0 for r in rs)))
    macro(f"nAllSeeds{k}", str(sum(int(r["seeds_positive"]) == int(r["J"]) for r in rs)))
for ds in MAIN:
    for t, k in (("tabpfn_3_5", "PFNThree"), ("tabicl_v2", "ICL"), ("tabpfn_2_5", "PFNTwo")):
        r = inf[(ds, "emp", t, "tuned")]
        macro(f"emp{k}{DS_KEY[ds]}", f"{num(r, 'rel_pct'):+.1f}")
        macro(f"empAbs{k}{DS_KEY[ds]}", f"{num(r, 'mean_diff'):+.3f}")
        macro(f"empLo{k}{DS_KEY[ds]}", f"{num(r, 'hier_lo'):+.3f}")
        macro(f"empHi{k}{DS_KEY[ds]}", f"{num(r, 'hier_hi'):+.3f}")
        macro(f"nbp{k}{DS_KEY[ds]}", pfmt(num(r, "nb_p")))
        macro(f"holm{k}{DS_KEY[ds]}", pfmt(num(r, "holm_p")))
        macro(f"seedsPos{k}{DS_KEY[ds]}", f"{r['seeds_positive']}")
        macro(f"iuHolm{k}{DS_KEY[ds]}", pfmt(num(r, "iu_holm_p")))
        macro(f"nbLo{k}{DS_KEY[ds]}", f"{num(r, 'nb_lo'):+.3f}")
        macro(f"nbHi{k}{DS_KEY[ds]}", f"{num(r, 'nb_hi'):+.3f}")

# secondary matrix: PR-AUC (points) and EMP (%) differences, markers as above
lines = []
for tfm in TFMS:
    lines.append(f"{M_NAME[tfm]} & " + " & ".join(cell(ds, met, tfm) for ds in MAIN for met in ("prauc", "emp")) + " \\\\")
SHORT = {"lgbm": "LGBM", "xgb": "XGB", "catboost": "CatB"}
refs = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\scriptsize vs.\\ {SHORT[inf[(ds, 'prauc', 'tabpfn_3_5', 'tuned')]['ref']]} / "
                  f"{SHORT[inf[(ds, 'emp', 'tabpfn_3_5', 'tuned')]['ref']]}" + "}" for ds in MAIN)
write("bootstrap.tex", "\n".join(lines) + "\n")
write("bootstrap_refs.tex", refs + " \\\\\n")
best_ref = {(ds, met): inf[(ds, met, "tabpfn_3_5", "tuned")]["ref"] for ds in MAIN + ["ecommerce"] for met in ("emp", "prauc")}

# ---------------------------------------------------------------- Table 4: rank agreement
ra = defaultdict(list)
for r in rows("rank_agreement.csv"):
    ra[(r["dataset"], r["metric"])].append(r)
lines = []
for ds in MAIN:
    out = [DS_NAME[ds]]
    for met in ("emp", "eprof_top20", "eprof_thr0.5_raw"):
        g = ra.get((ds, met))
        if not g:
            out += ["--", "--"]
            continue
        tau = st.mean(float(x["kendall_tau_vs_auc"]) for x in g)
        ch = sum(x["top1_changed"] == "True" for x in g)
        out += [f"${tau:.2f}$", f"{ch}/{len(g)}"]
        if met == "emp":
            macro(f"tauEMP{DS_KEY[ds]}", f"{tau:.2f}")
            macro(f"topEMP{DS_KEY[ds]}", f"{ch}")
        if met == "eprof_thr0.5_raw":
            macro(f"tauEPthr{DS_KEY[ds]}", f"{tau:.2f}")
    lines.append(" & ".join(out) + " \\\\")
write("rank_agreement.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table 5: calibration
reg = {(r["dataset"], r["model"]): r for r in rows("calibration_policy_regret.csv")}
lines = []
for ds in MAIN:
    gb = best_ref[(ds, "emp")]
    for i, m in enumerate([gb, "tabicl_v2", "tabpfn_2_5", "tabpfn_3_5"]):
        s, g = summ[(ds, m)], reg[(ds, m)]
        lines.append(("\\multirow{4}{*}{" + DS_NAME[ds] + "}" if i == 0 else "") +
                     f" & {M_NAME[m]} & ${100 * float(s['citl_raw_mean']):+.1f}$ & {f(s['ece_raw_mean'])} & {f(s['ece_iso_mean'])}"
                     f" & {100 * float(g['regret_raw']):.1f} & {100 * float(g['regret_iso']):.1f} & {100 * float(g['regret_platt']):.1f} \\\\")
    if ds != MAIN[-1]:
        lines.append("\\midrule")
write("calibration.tex", "\n".join(lines) + "\n")
for ds in MAIN:
    for m, k in (("tabicl_v2", "ICL"), ("tabpfn_3_5", "PFNThree"), ("tabpfn_2_5", "PFNTwo")):
        macro(f"citl{k}{DS_KEY[ds]}", f"{100 * float(summ[(ds, m)]['citl_raw_mean']):+.1f}")
macro("maxAbsCitlPFNThree", f"{max(abs(100 * float(summ[(ds, 'tabpfn_3_5')]['citl_raw_mean'])) for ds in MAIN):.1f}")
calm = [r for (ds, m), r in reg.items() if ds in MAIN and ds != "orange" and m in MODELS]
macro("maxRegretShiftNonOrange", f"{100 * max(max(abs(float(r['regret_raw']) - float(r['regret_iso'])), abs(float(r['regret_raw']) - float(r['regret_platt']))) for r in calm):.1f}")
macro("maxRawRegretNonOrange", f"{100 * max(float(r['regret_raw']) for r in calm):.1f}")
for k, col in (("Raw", "regret_raw"), ("Iso", "regret_iso"), ("Platt", "regret_platt")):
    macro(f"regICLOrange{k}", f"{100 * float(reg[('orange', 'tabicl_v2')][col]):.1f}")

# ---------------------------------------------------------------- Table 6: context ablation
ctx = {(r["dataset"], r["model"]): r for r in rows("context_ablation.csv")}
lines = []
for ds in ("orange", "cell2cell"):
    for base in ("tabpfn_3_5", "tabicl_v2"):
        for tag, label in (("", "full (natural rate)"), ("__ctx_1to2", "1:2"), ("__ctx_1to1", "1:1")):
            m = base + tag
            r = ctx[(ds, m)]
            b = inf[(ds, "emp", m, "tuned")]
            lines.append(f"{DS_NAME[ds]} & {M_NAME[base]} & {label} & {f(r['auc'])} & {f(r['prauc'])} & {f(r['emp'])} & "
                         f"${num(b, 'rel_pct'):+.1f}\\%$ & {b['seeds_positive']}/{b['J']} & {pfmt(num(b, 'nb_p'))} & "
                         f"${100 * float(r['citl_raw']):+.1f}$ \\\\")
            if ds == "orange" and base == "tabpfn_3_5":
                key = {"": "Full", "__ctx_1to2": "Half", "__ctx_1to1": "Bal"}[tag]
                macro("ctxOrangePFN" + key, f"{num(b, 'rel_pct'):+.1f}")
                macro("ctxOrangePFNp" + key, pfmt(num(b, "nb_p")))
        lines.append("\\midrule")
write("context.tex", "\n".join(lines[:-1]) + "\n")
macro("ctxCitlOrange", f"{100 * float(ctx[('orange', 'tabpfn_3_5__ctx_1to1')]['citl_raw']):+.1f}")

# ---------------------------------------------------------------- Table 7: replication on E-Commerce
ds = "ecommerce"
tuned = best_ref[(ds, "prauc")]
lines = []
for m in ["xgb_default", "xgb", "lgbm", "catboost", "tabicl_v2", "tabpfn_2_5", "tabpfn_3_5"]:
    s = summ[(ds, m)]
    if m in TFMS:
        dd = cell(ds, "prauc", m, "default")
        dt = cell(ds, "prauc", m, "tuned")
        de = cell(ds, "emp", m, "tuned")
    else:
        dd = dt = de = "--"
    if m == "tabicl_v2":
        lines.append("\\midrule")
    lines.append(f"{M_NAME[m]} & {f(s['prauc_mean'])} ({f(s['prauc_std'])}) & {f(s['emp_mean'])} & {dd} & {dt} & {de} \\\\")
write("replication.tex", "\n".join(lines) + "\n")
macro("replTuned", M_NAME[tuned])
for m, k in (("tabicl_v2", "ICL"), ("tabpfn_3_5", "PFNThree")):
    macro(f"repl{k}Default", f"{100 * num(inf[(ds, 'prauc', m, 'default')], 'mean_diff'):+.1f}")
    macro(f"repl{k}Tuned", f"{100 * num(inf[(ds, 'prauc', m, 'tuned')], 'mean_diff'):+.1f}")
    macro(f"repl{k}EMP", f"{num(inf[(ds, 'emp', m, 'tuned')], 'rel_pct'):+.1f}")
    macro(f"repl{k}TunedP", pfmt(num(inf[(ds, 'prauc', m, 'tuned')], 'nb_p')))
    macro(f"repl{k}Seeds", inf[(ds, 'prauc', m, 'tuned')]['seeds_positive'])
macro("replDefaultPR", f(summ[(ds, "xgb_default")]["prauc_mean"]))
macro("replTunedPR", f(summ[(ds, tuned)]["prauc_mean"]))

# ---------------------------------------------------------------- Table 9: sensitivity (tuning budget, duplicates)
lines = []
for ds in ("cell2cell", "iranian", "orange"):
    for tfm in ("tabpfn_3_5", "tabicl_v2"):
        a, b = inf.get((ds, "emp", tfm, "tuned")), inf.get((ds, "emp", tfm, "strong"))
        if not (a and b):
            continue
        lines.append(f"{DS_NAME[ds]} & {M_NAME[tfm]} & {M_NAME.get(a['ref'], a['ref'])} & ${num(a, 'rel_pct'):+.1f}\\%$ & "
                     f"{M_NAME.get(b['ref'], b['ref'])} & ${num(b, 'rel_pct'):+.1f}\\%$ & {b['seeds_positive']}/{b['J']} & "
                     f"{pfmt(num(b, 'nb_p'))} \\\\")
        if tfm == "tabpfn_3_5":
            macro(f"strongPFNThree{DS_KEY[ds]}", f"{num(b, 'rel_pct'):+.1f}")
            macro(f"strongRef{DS_KEY[ds]}", M_NAME.get(b["ref"], b["ref"]))
if lines:
    write("strong.tex", "\n".join(lines) + "\n")
per_seed = defaultdict(list)
for r in rows("metrics_per_seed.csv"):
    per_seed[(r["dataset"], r["model"])].append(r)
for ds in ("cell2cell", "iranian", "orange"):  # best GBDT EMP with 200 vs 25-40 trials, on the seeds both were run
    seeds = {r["seed"] for m in ("lgbm_200", "catboost_200") for r in per_seed[(ds, m)]}
    best = lambda ms: max(st.mean(float(r["emp"]) for r in per_seed[(ds, m)] if r["seed"] in seeds) for m in ms)
    b40, b200 = best(["lgbm", "xgb", "catboost"]), best(["lgbm_200", "catboost_200"])
    macro(f"strongDelta{DS_KEY[ds]}", f"{100 * (b200 - b40) / abs(b40):+.1f}")
lines = []
for ds in ("telco", "iranian", "ecommerce"):
    meta = K / f"{ds}-dupgroup-cpu/data_meta.json"
    if not meta.exists():
        continue
    dstat = json.load(open(meta))["duplicates"]
    macro(f"conflictGroups{DS_KEY[ds]}", str(dstat["n_conflict_groups"]))
    macro(f"conflictRows{DS_KEY[ds]}", str(dstat["n_conflict_rows"]))
    macro(f"churnBefore{DS_KEY[ds]}", f"{100 * dstat['churn_rate_before']:.1f}")
    macro(f"churnAfter{DS_KEY[ds]}", f"{100 * dstat['churn_rate_after_dedup']:.1f}")
    for tfm in ("tabpfn_3_5", "tabicl_v2"):
        a, b = inf.get((ds, "emp", tfm, "tuned")), inf.get((f"{ds}_dupgroup", "emp", tfm, "tuned"))
        if not (a and b):
            continue
        lines.append(f"{DS_NAME[ds]} & {dstat['n_dup']} & {dstat['n_conflict_groups']} ({dstat['n_conflict_rows']}) & "
                     f"{100 * dstat['churn_rate_before']:.1f} / {100 * dstat['churn_rate_after_dedup']:.1f} & {M_NAME[tfm]} & "
                     f"${num(a, 'rel_pct'):+.1f}\\%$ & ${num(b, 'rel_pct'):+.1f}\\%$ & {pfmt(num(b, 'nb_p'))} \\\\")
if lines:
    write("dupgroup.tex", "\n".join(lines) + "\n")
dmeta = [json.load(open(p))["duplicates"] for p in sorted(K.glob("*-dupgroup-cpu/data_meta.json"))]
if dmeta:
    macro("maxChurnShift", f"{max(abs(100 * (d['churn_rate_before'] - d['churn_rate_after_dedup'])) for d in dmeta):.1f}")
shifts = [abs(num(inf[(ds, "emp", t, "tuned")], "rel_pct") - num(inf[(f"{ds}_dupgroup", "emp", t, "tuned")], "rel_pct"))
          for ds in ("telco", "iranian", "ecommerce") for t in TFMS if (f"{ds}_dupgroup", "emp", t, "tuned") in inf]
if shifts:
    macro("maxDupShift", f"{max(shifts):.1f}")

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
for pdf in (A / "figures").glob("*.pdf"):
    shutil.copy(pdf, FIG / pdf.name)

# ---------------------------------------------------------------- Table 8: scoring throughput
T = json.load(open(K / "all-timing/timing.json")) + json.load(open(K / "orange-tabicl-timing/timing.json"))
tim = {}
for r in T:
    if r.get("status") == "failed":
        continue
    tim[(r["dataset"], r["model"], r["n_scored"])] = r
# the first Orange run of uncached TabICL v2 at n=50k fell back to 2k chunks after an OOM; the rerun supersedes it
lines = []
TM = [("lgbm_1000trees", "LightGBM (1,000 trees, CPU)"), ("tabicl_v2", "TabICL v2"), ("tabicl_v2_cache", "TabICL v2, KV cache"),
      ("tabpfn_3_5", "TabPFN-3.5"), ("tabpfn_3_5_cache", "TabPFN-3.5, cached context"), ("tabpfn_3_5_fast", "TabPFN-3.5-Fast"),
      ("tabpfn_2_5", "TabPFN-2.5")]
for key, label in TM:
    cells = []
    for ds in ("telco", "cell2cell", "orange"):
        got = [tim.get((ds, key, n)) for n in (10000, 50000)]
        got = [g for g in got if g]
        if not got:
            small = tim.get((ds, key, 1000))
            # only the 1,000-row batch was run; its per-customer time is an upper bound for larger batches
            cells.append(f"{small['s_per_10k']:,.0f}$^\\dagger$" if small else "--")
        else:
            cells.append(f"{min(g['s_per_10k'] for g in got):,.1f}")
    lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
write("timing.tex", "\n".join(lines) + "\n")
for key, k in (("tabpfn_3_5_cache", "PFNCache"), ("lgbm_1000trees", "LGBM"), ("tabicl_v2_cache", "ICLCache")):
    v = min(tim[("orange", key, n)]["s_per_10k"] for n in (10000, 50000) if ("orange", key, n) in tim)
    macro(f"time{k}Orange", f"{v:.1f}" if v < 10 else f"{v:.0f}")
for ds in ("telco", "cell2cell", "orange"):
    macro(f"ctx{DS_KEY[ds]}", f"{tim[(ds, 'tabpfn_3_5_cache', 10000)]['n_context']:,}")
macro("timePFNUncachedSmallOrange", f"{tim[('orange', 'tabpfn_3_5', 1000)]['s_per_10k']:,.0f}")
macro("hoursPFNCacheMillion", f"{min(tim[('orange', 'tabpfn_3_5_cache', n)]['s_per_10k'] for n in (10000, 50000)) * 100 / 3600:.2f}")

all_runs = [r for p in sorted(K.glob("*/runs.json")) for r in json.load(open(p))]
macro("nRuns", f"{sum(r['status'] == 'ok' for r in all_runs):,}")
for k, v in list(MACROS.items()):  # unsigned twin for prose where the wording carries the sign ("gains 17.8%")
    if v[:1] in "+-":
        MACROS[k + "Abs"] = v[1:]
(TAB / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(MACROS.items())))
print(f"wrote {len(list(TAB.glob('*.tex')))} table files and {len(MACROS)} macros to {TAB}")
for k in sorted(MACROS):
    print(f"  \\{k} = {MACROS[k]}")
