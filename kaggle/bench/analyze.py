"""Analysis stage: reads every prediction kernel's output and computes all metrics.

Ranking: AUC, PR-AUC, lift@10%.  Profit: EMP (Verbraken et al. 2013), e-Profits (Manzoor et al., KM-based CLV),
fixed-threshold campaign policy over a cost grid.  Calibration: Brier, ECE, raw vs isotonic vs Platt.
Statistics: mean/sd over seeds, paired bootstrap CIs of each TFM vs the best tuned GBDT, rank agreement between
metrics, and scoring-cost break-even.  Writes CSVs + summary.md to /kaggle/working.
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
                   prauc=average_precision_score(yte, pte), lift10=lift10(yte, pte), emp=emp(yte, pte))
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

    # ---- context-construction ablation vs the same model with its full natural-rate context
    abl = M[M.model.map(is_ablation)].assign(base=lambda d: d.model.str.split("__ctx_").str[0])
    ABL = pd.DataFrame()
    if len(abl):
        cols = ["auc", "prauc", "emp", "ece_raw", "citl_raw"]
        base = M[M.model.isin(abl.base.unique()) & M.dataset.isin(abl.dataset.unique())]
        ABL = pd.concat([base.assign(base=base.model), abl]).groupby(["dataset", "base", "model"])[cols].mean()
    ABL.to_csv(f"{OUT}/context_ablation.csv")

    # ---- rank agreement between ranking metrics and profit metrics (main models only)
    M_all, M = M, M[~M.model.map(is_ablation) & ~M.dataset.isin(REPLICATION) & (M.model != "xgb_default")]
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
        for title, df in [("Mean metrics (sd over seeds) + timing", agg), ("TFM vs best tuned GBDT (paired bootstrap 95% CI)", BOOT),
                          ("Context-construction ablation (mean over seeds)", ABL),
                          ("Calibration-in-the-large (observed - mean predicted churn rate)", CITL),
                          ("Average rank per metric (lower is better)", AVG_RANK), ("Rank agreement with AUC", RA.groupby(["dataset", "metric"])[["kendall_tau_vs_auc", "top1_changed"]].mean()),
                          ("Fixed-threshold policy regret vs oracle (median over cost grid & seeds)", CAL), ("Break-even (EMP units = currency with CLV=200)", BE)]:
            fh.write(f"\n## {title}\n\n{df.round(4).to_markdown()}\n")
    print(open(f"{OUT}/summary.md").read())


if __name__ == "__main__":
    main()
