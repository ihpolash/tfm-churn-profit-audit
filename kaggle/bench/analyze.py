"""Analysis stage: reads every prediction kernel's output and computes all metrics.

Ranking: AUC, PR-AUC, lift@10%.  Profit: EMP (Verbraken et al. 2013), e-Profits (Manzoor et al., KM-based CLV),
fixed-threshold campaign policy over a cost grid.  Calibration: Brier, ECE, raw vs isotonic vs Platt.
Statistics: mean/sd over seeds; per comparison of a TFM with the best tuned GBDT, the per-seed differences, a
Nadeau-Bengio corrected resampled t-test across seeds, a two-level (seed, then customer) bootstrap interval, and Holm
correction over the pre-declared primary family (EMP, four TFMs x five main datasets); the earlier pooled customer
bootstrap is kept for reference. Also rank agreement, calibration regret, break-even, a forest plot and a reliability
diagram. Writes CSVs, figures/ and summary.md to CHURN_OUTPUT (default /kaggle/working).
"""
import glob, json, os, warnings, zlib
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
# On Kaggle the prediction kernels are attached under /kaggle/input; locally, point CHURN_INPUT at results/kernels.
# Labels and e-Profits columns (data_side.npz) are not redistributed: run on Kaggle, or regenerate them with predict.py.
INPUT = os.environ.get("CHURN_INPUT", "/kaggle/input")
OUT = os.environ.get("CHURN_OUTPUT", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)
B = 500                       # bootstrap resamples per seed
CLV, D, F, A, BETA = 200.0, 10.0, 1.0, 6.0, 14.0  # EMP defaults (Verbraken et al. 2013)
GBAR = A / (A + BETA)
GPU_USD_PER_H = 0.35          # on-demand T4 price used for the break-even table
GBDT = ["lgbm", "xgb", "catboost"]
is_tfm = lambda m: m.startswith(("tabpfn", "tabicl"))
is_ablation = lambda m: "__ctx_" in m  # context-construction variants, kept out of the main rank tables
REPLICATION = {"ecommerce"}            # workshop-paper replication check, kept out of cross-dataset tables
STRONG = ["lgbm_200", "catboost_200"]  # 200-trial baselines, used only in the tuning-budget sensitivity check
BASE_TFMS = ["tabicl_v2", "tabpfn_2_5", "tabpfn_3_5", "tabpfn_3_5_fast"]
BH = 1000                              # two-level bootstrap resamples
is_sensitivity = lambda ds: ds.endswith("_dupgroup")  # duplicates kept, split by duplicate group
GAMMAS = (np.arange(200) + 0.5) / 200
W = stats.beta.pdf(GAMMAS, A, BETA); W /= W.sum()


# ---------------------------------------------------------------- metrics
def lift10(y, p):
    k = int(np.ceil(0.1 * len(y)))
    return y[np.argsort(-p)[:k]].mean() / y.mean()


def ece(y, p, nb=15):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(nb) if (b == k).any())


def gains(y, gamma, clv=CLV, d=D, f=F):
    return np.where(y == 1, gamma * (clv - d - f) - (1 - gamma) * f, -(d + f))


def emp_fast(y, p):
    """EMP via cumulative churner/non-churner counts; algebraically identical to emp(), several times faster."""
    ys = y[np.argsort(-p)]
    n1 = np.concatenate([[0], np.cumsum(ys)]); n0 = np.arange(len(ys) + 1) - n1
    prof = np.outer(GAMMAS * (CLV - D) - F, n1) - (D + F) * n0[None, :]
    return float((W * prof.max(axis=1)).sum() / len(y))


def emp(y, p):
    ys = y[np.argsort(-p)]
    g = np.where(ys[None, :] == 1, GAMMAS[:, None] * (CLV - D - F) - (1 - GAMMAS[:, None]) * F, -(D + F))
    best = np.maximum(np.cumsum(g, axis=1).max(axis=1), 0.0)
    return float((W * best).sum() / len(y))


def policy(y, p, clv, d):
    gc, gs = gains(np.array([1, 0]), GBAR, clv, d)
    t = p * gc + (1 - p) * gs > 0
    return float(gains(y[t], GBAR, clv, d).sum() / len(y))


def oracle(y, p, clv, d):
    return float(max(np.cumsum(gains(y[np.argsort(-p)], GBAR, clv, d)).max(), 0.0) / len(y))


def km_survival(tenure, event):
    """Kaplan-Meier S(t) as a step function evaluated by searchsorted."""
    t = np.unique(tenure[event == 1])
    at_risk = np.array([(tenure >= u).sum() for u in t])
    deaths = np.array([((tenure == u) & (event == 1)).sum() for u in t])
    s = np.cumprod(1 - deaths / at_risk)
    return lambda q: np.where(np.searchsorted(t, q, side="right") == 0, 1.0,
                              s[np.maximum(np.searchsorted(t, q, side="right") - 1, 0)])


def eprofits_values(tenure_tr, y_tr, tenure, revenue, margin=0.3, cpo=0.1):
    """Per-customer TP and FP payoffs of e-Profits (TRR variant, KM fitted on the training split)."""
    S = km_survival(tenure_tr, y_tr)
    r = np.clip(S(tenure + 1) / np.maximum(S(tenure), 1e-9), 0, 0.995)
    clv = revenue * margin / (1 - r)
    offer = cpo * clv
    contact = np.maximum(5.0, 0.3 * offer)
    return clv - offer - contact, -offer - contact


def eprofits(y, p, tp, fp, mode):
    if mode == "thr0.5":
        t = p >= 0.5
    elif mode == "top20":
        t = np.zeros(len(p), bool); t[np.argsort(-p)[:int(0.2 * len(p))]] = True
    else:  # best cutoff (oracle)
        o = np.argsort(-p)
        return float(max(np.cumsum(np.where(y[o] == 1, tp[o], fp[o])).max(), 0.0) / len(y))
    return float(np.where(y[t] == 1, tp[t], fp[t]).sum() / len(y))


def calibrators(pcal, ycal, pte):
    eps = 1e-6
    lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    return {"raw": pte,
            "iso": IsotonicRegression(out_of_bounds="clip").fit(pcal, ycal).predict(pte),
            "platt": LogisticRegression(C=1e6).fit(lg(pcal)[:, None], ycal).predict_proba(lg(pte)[:, None])[:, 1]}


_rng = np.random.default_rng(1); _y = (_rng.random(3000) < .2).astype(int); _p = _rng.random(3000)
assert abs(emp(_y, _p) - emp_fast(_y, _p)) < 1e-9, 'emp_fast disagrees with emp'


# ---------------------------------------------------------------- load
def load():
    runs, preds, side = [], {}, {}
    for rj in glob.glob(f"{INPUT}/**/runs.json", recursive=True):
        root = os.path.dirname(rj)
        meta = json.load(open(f"{root}/data_meta.json"))
        ds = meta["dataset"]
        sd = np.load(f"{root}/data_side.npz")
        side[ds] = {k: sd[k] for k in sd.files} | {"meta": meta}
        for r in json.load(open(rj)):
            runs.append(r)
            f = f"{root}/preds/{r['model']}_s{r['seed']}.npz"
            if r["status"] == "ok" and os.path.exists(f):
                preds[(ds, r["model"], r["seed"])] = dict(np.load(f))
    return pd.DataFrame(runs), preds, side


def main():
    runs, preds, side = load()
    print("runs:", runs.groupby(["dataset", "model", "status"]).size().to_string())
    rows, pol = [], []
    for (ds, model, seed), P in sorted(preds.items()):
        y = side[ds]["y"]; yte, ycal, ytr = y[P["idx_test"]], y[P["idx_cal"]], y[P["idx_train"]]
        pte = P["p_test"]
        row = dict(dataset=ds, model=model, seed=seed, auc=roc_auc_score(yte, pte),
                   prauc=average_precision_score(yte, pte), lift10=lift10(yte, pte), emp=emp(yte, pte),
                   # calibration-split scores: used only to choose the reference GBDT, never the test set
                   auc_cal=roc_auc_score(ycal, P["p_cal"]), prauc_cal=average_precision_score(ycal, P["p_cal"]),
                   emp_cal=emp(ycal, P["p_cal"]))
        cal = calibrators(P["p_cal"], ycal, pte)
        for k, p in cal.items():
            row[f"brier_{k}"] = brier_score_loss(yte, p); row[f"ece_{k}"] = ece(yte, p)
            row[f"citl_{k}"] = yte.mean() - p.mean()  # calibration-in-the-large: >0 means churn is underpredicted
        if side[ds]["tenure"].size:
            ten, rev = side[ds]["tenure"].astype(float), side[ds]["revenue"].astype(float)
            ok = ~np.isnan(ten[P["idx_train"]])
            tp, fp = eprofits_values(ten[P["idx_train"]][ok], ytr[ok], np.nan_to_num(ten[P["idx_test"]]),
                                     np.nan_to_num(rev[P["idx_test"]]))
            for k in ("raw", "iso"):
                row[f"eprof_thr0.5_{k}"] = eprofits(yte, cal[k], tp, fp, "thr0.5")
            row["eprof_top20"] = eprofits(yte, pte, tp, fp, "top20")
            row["eprof_best"] = eprofits(yte, pte, tp, fp, "best")
        rows.append(row)
        for clv in (100, 200, 500):
            for d in (5, 10, 20):
                rec = dict(dataset=ds, model=model, seed=seed, clv=clv, d=d, oracle=oracle(yte, pte, clv, d))
                for k, p in cal.items():
                    rec[f"profit_{k}"] = policy(yte, p, clv, d)
                pol.append(rec)
    M = pd.DataFrame(rows); POL = pd.DataFrame(pol)
    M.to_csv(f"{OUT}/metrics_per_seed.csv", index=False); POL.to_csv(f"{OUT}/policy_per_seed.csv", index=False)

    t = runs[runs.status == "ok"].groupby(["dataset", "model"])[["fit_s", "pred_s_per_10k", "gpu_peak_gb"]].mean()
    agg = M.groupby(["dataset", "model"]).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.drop(columns=[c for c in agg.columns if c.startswith("seed_")]).join(t)
    agg.to_csv(f"{OUT}/metrics_summary.csv")

    # ---- paired bootstrap: each TFM vs the best tuned GBDT (chosen per dataset and metric on mean over seeds)
    boot = []
    fns = {"auc": roc_auc_score, "prauc": average_precision_score, "emp": emp}
    for ds in M.dataset.unique():
        for met, fn in fns.items():
            mean = M[M.dataset == ds].groupby("model")[met].mean()
            gb = [m for m in GBDT if m in mean.index]
            if not gb:
                continue
            refs = [mean[gb].idxmax()] + [m for m in ("xgb_default",) if m in mean.index]
            for ref, tfm in [(r, m) for r in refs for m in mean.index if is_tfm(m)]:
                # one fixed stream per comparison, so adding rows elsewhere never changes these draws
                rng = np.random.default_rng(zlib.crc32(f"{ds}|{met}|{tfm}|{ref}".encode()))
                diffs = []
                for seed in sorted(M.seed.unique()):
                    if (ds, tfm, seed) not in preds or (ds, ref, seed) not in preds:
                        continue
                    a, b = preds[(ds, tfm, seed)], preds[(ds, ref, seed)]
                    assert np.array_equal(a["idx_test"], b["idx_test"]), "splits differ between kernels"
                    y = side[ds]["y"][a["idx_test"]]
                    for _ in range(B):
                        i = rng.integers(0, len(y), len(y))
                        if y[i].min() == y[i].max():
                            continue
                        diffs.append(fn(y[i], a["p_test"][i]) - fn(y[i], b["p_test"][i]))
                d = np.array(diffs)
                boot.append(dict(dataset=ds, metric=met, tfm=tfm, ref_gbdt=ref, diff_mean=d.mean(),
                                 ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                                 rel_diff_pct=100 * d.mean() / abs(mean[ref])))
    BOOT = pd.DataFrame(boot); BOOT.to_csv(f"{OUT}/bootstrap_vs_best_gbdt.csv", index=False)

    # ---- inference that treats the split as a source of variation
    INF = inference(M, preds, side)
    INF.to_csv(f"{OUT}/inference.csv", index=False)
    figures(INF, M, preds, side)

    # ---- context-construction ablation vs the same model with its full natural-rate context
    abl = M[M.model.map(is_ablation)].assign(base=lambda d: d.model.str.split("__ctx_").str[0])
    ABL = pd.DataFrame()
    if len(abl):
        cols = ["auc", "prauc", "emp", "ece_raw", "citl_raw"]
        base = M[M.model.isin(abl.base.unique()) & M.dataset.isin(abl.dataset.unique())]
        ABL = pd.concat([base.assign(base=base.model), abl]).groupby(["dataset", "base", "model"])[cols].mean()
    ABL.to_csv(f"{OUT}/context_ablation.csv")

    # ---- rank agreement between ranking metrics and profit metrics (main models only)
    M_all, M = M, M[~M.model.map(is_ablation) & ~M.dataset.isin(REPLICATION) & (M.model != "xgb_default")
                    & ~M.dataset.map(is_sensitivity) & ~M.model.isin(STRONG)]
    ra = []
    pm = [c for c in ("emp", "eprof_top20", "eprof_best", "eprof_thr0.5_raw") if c in M.columns]
    for (ds, seed), g in M.groupby(["dataset", "seed"]):
        for met in pm:
            if g[met].isna().all():
                continue
            tau = stats.kendalltau(g["auc"], g[met]).statistic
            ra.append(dict(dataset=ds, seed=seed, metric=met, kendall_tau_vs_auc=tau,
                           top1_auc=g.loc[g.auc.idxmax(), "model"], top1_metric=g.loc[g[met].idxmax(), "model"]))
    RA = pd.DataFrame(ra); RA["top1_changed"] = RA.top1_auc != RA.top1_metric
    RA.to_csv(f"{OUT}/rank_agreement.csv", index=False)
    ranks = pd.concat({met: M.groupby(["dataset", "seed"])[met].rank(ascending=False) for met in ["auc", "prauc", "emp"] + pm[1:]}, axis=1)
    AVG_RANK = ranks.join(M[["dataset", "model"]]).groupby("model").mean(numeric_only=True).sort_values("emp")

    # ---- calibration effect on fixed-threshold policy (fraction of oracle profit)
    for k in ("raw", "iso", "platt"):
        POL[f"regret_{k}"] = (POL.oracle - POL[f"profit_{k}"]) / POL.oracle.replace(0, np.nan)
    CAL = POL.groupby(["dataset", "model"])[["regret_raw", "regret_iso", "regret_platt"]].median()
    CAL.to_csv(f"{OUT}/calibration_policy_regret.csv")

    # ---- break-even: EMP gain per customer vs GPU scoring cost per customer
    be = []
    for r in BOOT[BOOT.metric == "emp"].itertuples():
        s10k = t.loc[(r.dataset, r.tfm), "pred_s_per_10k"]
        usd_per_cust = s10k / 1e4 / 3600 * GPU_USD_PER_H
        be.append(dict(dataset=r.dataset, tfm=r.tfm, ref=r.ref_gbdt, emp_gain_per_cust=r.diff_mean,
                       emp_gain_ci=(round(r.ci_lo, 4), round(r.ci_hi, 4)), t4_hours_per_1M=s10k * 100 / 3600,
                       usd_per_1M_scored=usd_per_cust * 1e6, gain_per_1M=r.diff_mean * 1e6,
                       gpu_cost_share_of_gain=usd_per_cust / r.diff_mean if r.diff_mean > 0 else np.nan))
    BE = pd.DataFrame(be); BE.to_csv(f"{OUT}/break_even.csv", index=False)

    CITL = M_all.groupby(["dataset", "model"])[["citl_raw", "citl_iso", "citl_platt"]].mean()

    with open(f"{OUT}/summary.md", "w") as fh:
        for title, df in [("Mean metrics (sd over seeds) + timing", agg),
                          ("Primary inference: corrected t-test across seeds, Holm, two-level bootstrap", INF),
                          ("TFM vs best tuned GBDT (pooled customer bootstrap, reference only)", BOOT),
                          ("Context-construction ablation (mean over seeds)", ABL),
                          ("Calibration-in-the-large (observed - mean predicted churn rate)", CITL),
                          ("Average rank per metric (lower is better)", AVG_RANK), ("Rank agreement with AUC", RA.groupby(["dataset", "metric"])[["kendall_tau_vs_auc", "top1_changed"]].mean()),
                          ("Fixed-threshold policy regret vs oracle (median over cost grid & seeds)", CAL), ("Break-even (EMP units = currency with CLV=200)", BE)]:
            fh.write(f"\n## {title}\n\n{df.round(4).to_markdown()}\n")
    print(open(f"{OUT}/summary.md").read())


def inference(M, preds, side):
    """Per comparison: per-seed differences on the full test set, Nadeau-Bengio corrected t-test over seeds
    (variance inflated by 1/J + n_test/n_train for overlapping random splits), and a two-level bootstrap that
    resamples seeds and then customers. The reference GBDT is chosen on the calibration split ("tuned"); the
    test-selected reference ("tuned_test") and each tuned GBDT separately ("vs_<model>", combined by an
    intersection-union test) are sensitivity checks. Holm is applied within each family of 20 comparisons."""
    fns = {"emp": (emp, emp_fast), "prauc": (average_precision_score,) * 2, "auc": (roc_auc_score,) * 2}
    rows = []
    for ds in M.dataset.unique():
        mean = {met: M[M.dataset == ds].groupby("model")[met].mean() for met in fns}
        mean_cal = {met: M[M.dataset == ds].groupby("model")[f"{met}_cal"].mean() for met in fns}
        for met, (fn, fast) in fns.items():
            gb = [m for m in GBDT if m in mean[met].index]
            if not gb:
                continue
            refsets = {"tuned": mean_cal[met][gb].idxmax(), "tuned_test": mean[met][gb].idxmax()}
            refsets.update({f"vs_{m}": m for m in gb})
            only = {}  # refset -> seeds it is restricted to
            strong = [m for m in STRONG if m in mean[met].index]
            if strong:  # tuning-budget check: both references chosen and compared on the splits the 200-trial runs cover
                ss = set(M[(M.dataset == ds) & M.model.isin(strong)].seed)
                cal_s = M[(M.dataset == ds) & M.seed.isin(ss)].groupby("model")[f"{met}_cal"].mean()
                refsets["strong_base"], refsets["strong"] = cal_s[gb].idxmax(), cal_s[gb + strong].idxmax()
                only["strong_base"] = only["strong"] = ss
            if "xgb_default" in mean[met].index:
                refsets["default"] = "xgb_default"
            for refset, ref in refsets.items():
                for tfm in [m for m in mean[met].index if is_tfm(m)]:
                    seeds = sorted(s for (d, m, s) in preds if d == ds and m == tfm and (ds, ref, s) in preds
                                   and s in only.get(refset, [s]))
                    if len(seeds) < 2:
                        continue
                    cells = []
                    for s in seeds:
                        a, b = preds[(ds, tfm, s)], preds[(ds, ref, s)]
                        assert np.array_equal(a["idx_test"], b["idx_test"])
                        cells.append((side[ds]["y"][a["idx_test"]], a["p_test"], b["p_test"],
                                      len(a["idx_test"]) / len(a["idx_train"])))
                    d = np.array([fn(y, pa) - fn(y, pb) for y, pa, pb, _ in cells])
                    J, ratio = len(d), np.mean([c[3] for c in cells])
                    var = (1 / J + ratio) * d.var(ddof=1)
                    tstat = d.mean() / np.sqrt(var) if var > 0 else np.inf * np.sign(d.mean())
                    pval = float(2 * stats.t.sf(abs(tstat), J - 1))
                    half = stats.t.ppf(0.975, J - 1) * np.sqrt(var)  # corrected-test 95% CI (unadjusted)
                    row = dict(dataset=ds, metric=met, tfm=tfm, ref=ref, ref_set=refset, J=J, ratio=ratio,
                               mean_diff=d.mean(), rel_pct=100 * d.mean() / abs(np.mean([fn(y, pb) for y, _, pb, _ in cells])),
                               seeds_positive=int((d > 0).sum()), nb_t=tstat, nb_p=pval,
                               nb_lo=d.mean() - half, nb_hi=d.mean() + half, **{f"diff_s{s}": v for s, v in zip(seeds, d)})
                    if met in ("emp", "prauc") and not is_ablation(tfm) and refset in ("tuned", "default"):
                        rng = np.random.default_rng(zlib.crc32(f"{ds}|{met}|{tfm}|{ref}|hier".encode()))
                        H = np.empty(BH)
                        for h in range(BH):
                            acc = 0.0
                            for j in rng.integers(0, J, J):
                                y, pa, pb, _ = cells[j]
                                i = rng.integers(0, len(y), len(y))
                                acc += fast(y[i], pa[i]) - fast(y[i], pb[i])
                            H[h] = acc / J
                        row.update(hier_lo=np.percentile(H, 2.5), hier_hi=np.percentile(H, 97.5))
                        for s, (y, pa, pb, _) in zip(seeds, cells):  # per-seed customer bootstrap interval
                            ps = [fast(y[i], pa[i]) - fast(y[i], pb[i])
                                  for i in (rng.integers(0, len(y), len(y)) for _ in range(300))]
                            row[f"lo_s{s}"], row[f"hi_s{s}"] = np.percentile(ps, [2.5, 97.5])
                    rows.append(row)
    INF = pd.DataFrame(rows)
    family = ((INF.metric == "emp") & INF.tfm.isin(BASE_TFMS) & ~INF.dataset.isin(REPLICATION)
              & ~INF.dataset.map(is_sensitivity))
    INF["primary"] = family & (INF.ref_set == "tuned")
    INF.loc[INF.primary, "holm_p"] = holm(INF.loc[INF.primary, "nb_p"])
    test_sel = family & (INF.ref_set == "tuned_test")  # sensitivity: reference chosen on test means
    INF.loc[test_sel, "holm_p"] = holm(INF.loc[test_sel, "nb_p"])
    # intersection-union: a TFM beats the tuned GBDTs only if it beats each of them; p = max over the three
    vs = INF[family & INF.ref_set.str.startswith("vs_")]
    worst = vs.loc[vs.groupby(["dataset", "tfm"]).nb_p.idxmax()].set_index(["dataset", "tfm"])
    keys = list(zip(INF.dataset, INF.tfm))
    INF.loc[INF.primary, "iu_p"] = [worst.loc[k, "nb_p"] for k, prim in zip(keys, INF.primary) if prim]
    INF.loc[INF.primary, "iu_ref"] = [worst.loc[k, "ref"] for k, prim in zip(keys, INF.primary) if prim]
    INF.loc[INF.primary, "iu_holm_p"] = holm(INF.loc[INF.primary, "iu_p"])
    return INF


def holm(p):
    """Holm step-down adjusted p-values for a pandas Series, returned in the original order."""
    order = p.sort_values()
    m = len(order)
    adj = np.minimum(1, np.maximum.accumulate([(m - k) * v for k, v in enumerate(order.values)]))
    return pd.Series(adj, index=order.index).reindex(p.index)


def figures(INF, M, preds, side):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    names = {"tabicl_v2": "TabICL v2", "tabpfn_2_5": "TabPFN-2.5", "tabpfn_3_5": "TabPFN-3.5",
             "tabpfn_3_5_fast": "TabPFN-3.5-Fast", "lgbm": "LightGBM", "xgb": "XGBoost", "catboost": "CatBoost"}
    order = ["telco", "maven", "cell2cell", "iranian", "orange"]
    P = INF[INF.primary].copy()
    P["o"] = P.dataset.map({d: i for i, d in enumerate(order)})
    P = P.sort_values(["o", "tfm"])
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    yt, yl = [], []
    for k, r in enumerate(P.itertuples()):
        yy = -k - 0.6 * r.o
        filled = r.holm_p < 0.05
        ax.errorbar(r.mean_diff, yy, xerr=[[r.mean_diff - r.nb_lo], [r.nb_hi - r.mean_diff]], fmt="o",
                    color="#1f4e79", mfc="#1f4e79" if filled else "white", ms=5, capsize=2, lw=1)
        yt.append(yy); yl.append(f"{r.dataset.capitalize() if r.dataset != 'cell2cell' else 'Cell2Cell'} · {names[r.tfm]}")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7)
    ax.set_xlabel("EMP difference to best tuned GBDT (per customer, CLV units)")
    ax.set_title("Filled: Holm-adjusted p < 0.05; bars: corrected-test 95% CI (unadjusted)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/forest_emp.pdf"); plt.close(fig)

    ds = "orange"
    gb = M[(M.dataset == ds) & M.model.isin(GBDT)].groupby("model").emp_cal.mean().idxmax()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, model in zip(axes, [gb, "tabicl_v2", "tabpfn_3_5"]):
        for kind, style in (("raw", "o-"), ("isotonic", "s--")):
            ys, ps = [], []
            for s in sorted(M.seed.unique()):
                P_ = preds.get((ds, model, s))
                if P_ is None:
                    continue
                y = side[ds]["y"]; yte, ycal = y[P_["idx_test"]], y[P_["idx_cal"]]
                p = P_["p_test"] if kind == "raw" else IsotonicRegression(out_of_bounds="clip").fit(P_["p_cal"], ycal).predict(P_["p_test"])
                ys.append(yte); ps.append(p)
            ys, ps = np.concatenate(ys), np.concatenate(ps)
            edges = np.unique(np.quantile(ps, np.linspace(0, 1, 11)))
            b = np.clip(np.searchsorted(edges, ps, side="right") - 1, 0, len(edges) - 2)
            xs = [ps[b == k].mean() for k in range(len(edges) - 1) if (b == k).any()]
            obs = [ys[b == k].mean() for k in range(len(edges) - 1) if (b == k).any()]
            ax.plot(xs, obs, style, ms=3, lw=1, label=kind)
        lim = 0.35
        ax.plot([0, lim], [0, lim], color="grey", lw=0.8)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title(names[model], fontsize=9)
        ax.set_xlabel("mean predicted churn", fontsize=8)
    axes[0].set_ylabel("observed churn rate", fontsize=8); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/reliability_orange.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
