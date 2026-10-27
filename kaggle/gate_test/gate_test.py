"""Stop-or-continue gate test for the TFM churn profit/calibration audit.

G1 feasibility: TabPFN and TabICL score Orange KDD09 (full ~30k-row context, ~200 features) on a T4.
G2 calibration matters: post-hoc calibration moves fixed-threshold campaign profit by >=5% of the
   oracle (best-cutoff) profit for at least one model/dataset, consistently across seeds.
Also reported: AUC, PR-AUC, lift@10%, EMP, Brier, ECE, timing, and whether EMP reorders models vs AUC.
"""
import glob, json, os, subprocess, sys, time, traceback, warnings

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn", "tabicl"], check=False)
for _tok in glob.glob("/kaggle/input/**/tabpfn_token.txt", recursive=True)[:1]:
    os.environ["TABPFN_TOKEN"] = open(_tok).read().strip()
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

OUT = "/kaggle/working"
RESULTS = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save():
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)


# ---------------------------------------------------------------- data
def to_numeric_frame(X):
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    X = X.astype("float32")
    keep = [c for c in X.columns if X[c].notna().mean() > 0.01 and X[c].nunique(dropna=True) > 1]
    return X[keep].values


def load_telco():
    f = glob.glob("/kaggle/input/**/WA_Fn-UseC_-Telco-Customer-Churn.csv", recursive=True)[0]
    df = pd.read_csv(f)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df.pop("Churn") == "Yes").astype(int).values
    return to_numeric_frame(df.drop(columns=["customerID"])), y


def load_orange():
    from sklearn.datasets import fetch_openml
    try:
        d = fetch_openml(data_id=1112, as_frame=True, parser="auto")
    except Exception:
        d = fetch_openml(name="KDDCup09_churn", version=1, as_frame=True, parser="auto")
    t = d.target.astype(str)
    log("orange target counts", t.value_counts().to_dict())
    y = t.isin(["1", "1.0", "True", "yes"]).astype(int).values
    return to_numeric_frame(d.data), y


# ---------------------------------------------------------------- metrics
CLV, D, F, A, B = 200.0, 10.0, 1.0, 6.0, 14.0  # Verbraken et al. 2013 defaults
GBAR = A / (A + B)


def ece(y, p, nb=15):
    bins = np.minimum((p * nb).astype(int), nb - 1)
    return sum((bins == k).mean() * abs(y[bins == k].mean() - p[bins == k].mean()) for k in range(nb) if (bins == k).any())


def lift10(y, p):
    k = int(np.ceil(0.1 * len(y)))
    return y[np.argsort(-p)[:k]].mean() / y.mean()


def gains(y, gamma, clv, d, f):
    return np.where(y == 1, gamma * (clv - d - f) - (1 - gamma) * f, -(d + f))


def profit_curve(y, p, gamma, clv=CLV, d=D, f=F):
    return np.concatenate([[0.0], np.cumsum(gains(y[np.argsort(-p)], gamma, clv, d, f))]) / len(y)


def emp(y, p, n=200):
    g = (np.arange(n) + 0.5) / n
    w = stats.beta.pdf(g, A, B)
    w /= w.sum()
    return float(sum(wi * profit_curve(y, p, gi).max() for gi, wi in zip(g, w)))


def policy_profit(y, p, clv, d):
    """Target every customer whose predicted expected profit is positive (uses p as a probability)."""
    g_churn, g_stay = gains(np.array([1, 0]), GBAR, clv, d, F)
    target = p * g_churn + (1 - p) * g_stay > 0
    return float(gains(y[target], GBAR, clv, d, F).sum() / len(y)), float(target.mean())


# ---------------------------------------------------------------- models
def fit_lgbm(X, y, seed):
    Xa, Xv, ya, yv = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    rng, best = np.random.RandomState(seed), (-1, None)
    for _ in range(20):
        m = lgb.LGBMClassifier(
            n_estimators=3000, learning_rate=10 ** rng.uniform(-2.3, -1), num_leaves=int(2 ** rng.uniform(3, 7)),
            min_child_samples=int(10 ** rng.uniform(0.7, 2.3)), subsample=rng.uniform(0.5, 1), subsample_freq=1,
            colsample_bytree=rng.uniform(0.3, 1), reg_lambda=10 ** rng.uniform(-3, 1), verbose=-1, random_state=seed)
        m.fit(Xa, ya, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        a = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
        if a > best[0]:
            best = (a, m)
    return best[1]


def fit_tabpfn(X, y, seed):
    from tabpfn import TabPFNClassifier
    try:
        m = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=True)
    except TypeError:
        m = TabPFNClassifier(device="cuda", random_state=seed)
    return m.fit(X, y)


def fit_tabicl(X, y, seed):
    from tabicl import TabICLClassifier
    return TabICLClassifier(device="cuda", random_state=seed).fit(X, y)


MODELS = {"lgbm_tuned": fit_lgbm, "tabicl": fit_tabicl, "tabpfn": fit_tabpfn}


def predict(m, X, bs=2000):
    return np.concatenate([m.predict_proba(X[i:i + bs])[:, 1] for i in range(0, len(X), bs)])


def describe(m):
    return {k: str(getattr(m, k)) for k in ("model_path", "checkpoint_version", "model_version") if hasattr(m, k)}


# ---------------------------------------------------------------- run
def run(ds, X, y, seeds):
    log(f"== {ds}: X={X.shape} churn={y.mean():.3f}")
    for seed in seeds:
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
        Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed)
        for name, fit in MODELS.items():
            row = dict(dataset=ds, seed=seed, model=name, n_train=len(ytr), n_feat=X.shape[1])
            Xfit, yfit = Xtr, ytr
            for attempt in ("full", "sub10k", "imputed"):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    t0 = time.time(); m = fit(Xfit, yfit, seed); row["fit_s"] = time.time() - t0
                    t0 = time.time(); pte = predict(m, Xte); row["pred_s_per_10k"] = (time.time() - t0) / len(yte) * 1e4
                    pcal = predict(m, Xcal)
                    row.update(status="ok", context_used=len(yfit), attempt=attempt, info=describe(m),
                               gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0)
                    break
                except Exception as e:
                    msg = f"{type(e).__name__}: {str(e)[:300]}"
                    log(f"   {name} seed{seed} attempt={attempt} FAILED {msg if 'errors' in row else str(e)}")
                    row.setdefault("errors", []).append(msg)
                    if attempt == "full" and len(ytr) > 10000:
                        Xfit, _, yfit, _ = train_test_split(Xtr, ytr, train_size=10000, stratify=ytr, random_state=seed)
                    else:
                        med = np.nanmedian(Xtr, axis=0)
                        Xfit, Xte, Xcal = (np.where(np.isnan(a), med, a) for a in (Xfit, Xte, Xcal))
            if row.get("status") != "ok":
                row["status"] = "failed"; RESULTS.append(row); save(); continue

            eps = 1e-6
            lc, lt = (np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))) for p in (pcal, pte))
            calibrated = {
                "raw": pte,
                "isotonic": IsotonicRegression(out_of_bounds="clip").fit(pcal, ycal).predict(pte),
                "platt": LogisticRegression(C=1e6).fit(lc[:, None], ycal).predict_proba(lt[:, None])[:, 1],
            }
            row.update(auc=roc_auc_score(yte, pte), prauc=average_precision_score(yte, pte), lift10=lift10(yte, pte),
                       emp=emp(yte, pte))
            for k, p in calibrated.items():
                row[f"brier_{k}"] = brier_score_loss(yte, p); row[f"ece_{k}"] = ece(yte, p)
            row["policy"] = []
            for clv in (100, 200, 500):
                for d in (5, 10, 20):
                    oracle = float(profit_curve(yte, pte, GBAR, clv, d).max())
                    rec = dict(clv=clv, d=d, oracle=oracle)
                    for k, p in calibrated.items():
                        rec[f"profit_{k}"], rec[f"frac_{k}"] = policy_profit(yte, p, clv, d)
                    row["policy"].append(rec)
            log(f"   {name} seed{seed} ctx={row['context_used']} fit={row['fit_s']:.0f}s auc={row['auc']:.4f} "
                f"prauc={row['prauc']:.4f} emp={row['emp']:.3f} ece_raw={row['ece_raw']:.4f} ece_iso={row['ece_isotonic']:.4f}")
            RESULTS.append(row); save()


def summarize():
    ok = [r for r in RESULTS if r["status"] == "ok"]
    df = pd.DataFrame(ok)
    print("\n===== RANKING / PROFIT / CALIBRATION (mean over seeds) =====")
    cols = ["auc", "prauc", "lift10", "emp", "brier_raw", "brier_isotonic", "ece_raw", "ece_isotonic", "ece_platt",
            "fit_s", "pred_s_per_10k", "gpu_peak_gb", "context_used"]
    print(df.groupby(["dataset", "model"])[cols].mean().round(4).to_string())

    pol = pd.DataFrame([dict(dataset=r["dataset"], model=r["model"], seed=r["seed"], **p) for r in ok for p in r["policy"]])
    for k in ("raw", "isotonic", "platt"):
        pol[f"regret_{k}"] = (pol["oracle"] - pol[f"profit_{k}"]) / pol["oracle"].abs()
    pol["cal_gain_iso"] = (pol["profit_isotonic"] - pol["profit_raw"]) / pol["oracle"].abs()
    pol["cal_gain_platt"] = (pol["profit_platt"] - pol["profit_raw"]) / pol["oracle"].abs()
    print("\n===== FIXED-THRESHOLD POLICY: regret vs oracle cutoff, and gain from calibration (fraction of oracle) =====")
    agg = pol.groupby(["dataset", "model", "clv", "d"])[["oracle", "regret_raw", "regret_isotonic", "regret_platt",
                                                      "cal_gain_iso", "cal_gain_platt", "frac_raw", "frac_isotonic"]].mean()
    print(agg.round(3).to_string())
    pol.to_csv(f"{OUT}/policy.csv", index=False)

    print("\n===== RANK ORDER: AUC vs EMP per dataset/seed =====")
    for (ds, s), g in df.groupby(["dataset", "seed"]):
        print(ds, s, "AUC:", list(g.sort_values("auc", ascending=False).model), "| EMP:", list(g.sort_values("emp", ascending=False).model))

    print("\n===== GATES =====")
    full = {(r["dataset"], r["model"]): r["context_used"] == r["n_train"] for r in ok}
    for m in ("tabpfn", "tabicl"):
        st = [r for r in RESULTS if r["dataset"] == "orange" and r["model"] == m]
        print(f"G1 {m} on orange:", [(r["status"], r.get("context_used"), round(r.get("fit_s", 0)), round(r.get("gpu_peak_gb", 0), 1)) for r in st],
              "full-context" if full.get(("orange", m)) else "NOT full-context")
    best = pol.groupby(["dataset", "model", "clv", "d"]).agg(
        gain=("cal_gain_iso", "mean"), gain_platt=("cal_gain_platt", "mean"),
        min_seed=("cal_gain_iso", "min"), max_seed=("cal_gain_iso", "max"))
    best["abs"] = best[["gain", "gain_platt"]].abs().max(axis=1)
    print("G2 top calibration effects (|gain| as fraction of oracle profit):")
    print(best.sort_values("abs", ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    log("TABPFN_TOKEN loaded:", bool(os.environ.get("TABPFN_TOKEN")))
    log("torch", torch.__version__, "cuda", torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    for p in ("tabpfn", "tabicl", "lightgbm"):
        try:
            from importlib.metadata import version
            log(p, version(p))
        except Exception as e:
            log(p, "import failed", e)
    for ds, loader, seeds in (("telco", load_telco, [0, 1, 2]), ("orange", load_orange, [0, 1])):
        try:
            X, y = loader()
            run(ds, X, y, seeds)
        except Exception:
            log(f"dataset {ds} crashed"); traceback.print_exc()
    save()
    summarize()
