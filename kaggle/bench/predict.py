"""Prediction stage of the churn TFM benchmark.

One kernel per (dataset, group). Writes test/calibration probabilities for every (model, seed) to
/kaggle/working/preds/<model>_s<seed>.npz plus timing in runs.json. All metrics are computed later by
analyze.py, so they can change without rerunning models. Splits are deterministic per seed (60/20/20).
"""
import glob, json, os, subprocess, sys, time, traceback, warnings
from functools import partial

DATASET = "__DATASET__"  # telco | maven | cell2cell | iranian | orange | ecommerce (replication) | all (timing)
                         # suffix _dupgroup: keep duplicates, but never split identical rows (sensitivity check)
GROUP = "__GROUP__"      # cpu | gpu | context | strong (200-trial GBDTs) | timing
GPU = GROUP in ("gpu", "context", "timing")
BASE = DATASET.removesuffix("_dupgroup")
DEDUP = not DATASET.endswith("_dupgroup")
SEEDS = [0, 1, 2, 3, 4]
ALL_DATASETS = ["telco", "maven", "cell2cell", "iranian", "orange"]

pkgs = ["tabpfn", "tabicl"] if GPU else ["interpret", "optuna", "catboost"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
# TabPFN weights need a licence-accepted token (https://ux.priorlabs.ai): a Kaggle Secret named TABPFN_TOKEN,
# or a file tabpfn_token.txt in an attached private dataset.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ.setdefault("TABPFN_TOKEN", UserSecretsClient().get_secret("TABPFN_TOKEN"))
except Exception:
    pass
for _tok in glob.glob("/kaggle/input/**/tabpfn_token.txt", recursive=True)[:1]:
    os.environ["TABPFN_TOKEN"] = open(_tok).read().strip()
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OUT = "/kaggle/working"
os.makedirs(f"{OUT}/preds", exist_ok=True)
RUNS = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- data
def find(name):
    return glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]


def finish(df, y, tenure=None, revenue=None):
    """Ordinal-encode categoricals, drop empty/constant columns; keep tenure/revenue for e-Profits."""
    side = {"tenure": None if tenure is None else pd.to_numeric(df[tenure], errors="coerce").values,
            "revenue": None if revenue is None else pd.to_numeric(df[revenue], errors="coerce").values}
    X, cat = df.copy(), []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
            cat.append(c)
    X = X.astype("float32")
    keep = [c for c in X.columns if X[c].notna().mean() > 0.01 and X[c].nunique(dropna=True) > 1]
    # Exact duplicate feature rows would let in-context learners look test rows up in their context: keep the first.
    y = np.asarray(y, dtype=int)
    group = pd.util.hash_pandas_object(X[keep], index=False).values  # identical feature rows share a group
    dup = pd.Series(group).duplicated().values
    labels_per_group = pd.Series(y).groupby(group).nunique()
    conflict = labels_per_group[labels_per_group > 1].index
    stats = {"n_dup": int(dup.sum()), "n_conflict_groups": int(len(conflict)),
             "n_conflict_rows": int(np.isin(group, conflict).sum()), "churn_rate_before": float(y.mean()),
             "churn_rate_after_dedup": float(y[~dup].mean())}
    if not DEDUP:  # keep duplicates; main() splits by group so identical rows never cross the split
        return X[keep].values, y, [keep.index(c) for c in cat if c in keep], keep, side | stats | {"group": group}
    side = {k: (None if v is None else v[~dup]) for k, v in side.items()} | stats
    return X[keep].values[~dup], y[~dup], [keep.index(c) for c in cat if c in keep], keep, side


def load_telco():
    df = pd.read_csv(find("WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df.pop("Churn") == "Yes").values
    return finish(df.drop(columns=["customerID"]), y, "tenure", "MonthlyCharges")


def load_maven():
    df = pd.read_csv(find("telecom_customer_churn.csv"))
    df = df[df["Customer Status"] != "Joined"]  # new customers have no churn outcome yet
    y = (df["Customer Status"] == "Churned").values
    df = df.drop(columns=["Customer ID", "Customer Status", "Churn Category", "Churn Reason"])
    return finish(df, y, "Tenure in Months", "Monthly Charge")


def load_cell2cell():
    df = pd.read_csv(find("cell2celltrain.csv"))
    y = (df.pop("Churn") == "Yes").values
    return finish(df.drop(columns=["CustomerID"]), y, "MonthsInService", "MonthlyRevenue")


def load_iranian():
    df = pd.read_csv(find("Customer Churn.csv"))
    df.columns = [c.strip() for c in df.columns]
    y = df.pop("Churn").values
    return finish(df, y)


def load_ecommerce():
    """Replication check: the workshop paper's headline dataset (CC-BY-NC-SA-4.0). No revenue column, so no e-Profits."""
    sheets = pd.read_excel(find("E Commerce Dataset.xlsx"), sheet_name=None)
    df = next(s for s in sheets.values() if "Churn" in s.columns)
    y = df.pop("Churn").values
    return finish(df.drop(columns=["CustomerID"]), y)


def load_orange():
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=1112, as_frame=True, parser="auto")
    return finish(d.data, (d.target.astype(str) == "1").values)


# ---------------------------------------------------------------- CPU models (tuned on an inner split, refit on full train)
def inner(X, y, seed):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def tune(objective, n_trials, seed, timeout=1200):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective, n_trials=n_trials, timeout=timeout)
    return st.best_trial


def fit_lgbm(X, y, seed, cat, n_trials=40, timeout=1200):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    Xa, Xv, ya, yv = inner(X, y, seed)

    def obj(t):
        p = dict(learning_rate=t.suggest_float("lr", 5e-3, 0.1, log=True), num_leaves=t.suggest_int("leaves", 8, 128, log=True),
                 min_child_samples=t.suggest_int("mcs", 5, 200, log=True), subsample=t.suggest_float("ss", 0.5, 1.0),
                 colsample_bytree=t.suggest_float("cs", 0.3, 1.0), reg_lambda=t.suggest_float("l2", 1e-3, 10, log=True))
        m = lgb.LGBMClassifier(n_estimators=3000, subsample_freq=1, verbose=-1, random_state=seed, **p)
        m.fit(Xa, ya, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        t.set_user_attr("n", m.best_iteration_ or 3000)
        return roc_auc_score(yv, m.predict_proba(Xv)[:, 1])

    b = tune(obj, n_trials, seed, timeout)
    p = b.params
    return lgb.LGBMClassifier(n_estimators=b.user_attrs["n"], subsample_freq=1, verbose=-1, random_state=seed,
                              learning_rate=p["lr"], num_leaves=p["leaves"], min_child_samples=p["mcs"], subsample=p["ss"],
                              colsample_bytree=p["cs"], reg_lambda=p["l2"]).fit(X, y)


def fit_xgb(X, y, seed, cat):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    Xa, Xv, ya, yv = inner(X, y, seed)

    def params(t):
        return dict(learning_rate=t.suggest_float("lr", 5e-3, 0.1, log=True), max_depth=t.suggest_int("depth", 3, 10),
                    min_child_weight=t.suggest_float("mcw", 0.5, 50, log=True), subsample=t.suggest_float("ss", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("cs", 0.3, 1.0), reg_lambda=t.suggest_float("l2", 1e-3, 10, log=True))

    def obj(t):
        m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=100, eval_metric="auc", tree_method="hist",
                              random_state=seed, n_jobs=-1, **params(t))
        m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        t.set_user_attr("n", m.best_iteration + 1)
        return roc_auc_score(yv, m.predict_proba(Xv)[:, 1])

    b = tune(obj, 40, seed)
    p = b.params
    return xgb.XGBClassifier(n_estimators=b.user_attrs["n"], tree_method="hist", random_state=seed, n_jobs=-1,
                             learning_rate=p["lr"], max_depth=p["depth"], min_child_weight=p["mcw"], subsample=p["ss"],
                             colsample_bytree=p["cs"], reg_lambda=p["l2"]).fit(X, y)


def fit_catboost(X, y, seed, cat, n_trials=25, timeout=1200):
    from catboost import CatBoostClassifier
    Xa, Xv, ya, yv = inner(X, y, seed)

    def params(t):
        return dict(learning_rate=t.suggest_float("lr", 1e-2, 0.2, log=True), depth=t.suggest_int("depth", 4, 10),
                    l2_leaf_reg=t.suggest_float("l2", 1, 30, log=True), random_strength=t.suggest_float("rs", 0.1, 10, log=True),
                    bagging_temperature=t.suggest_float("bt", 0, 1))

    def obj(t):
        m = CatBoostClassifier(iterations=3000, eval_metric="AUC", od_type="Iter", od_wait=100, random_seed=seed,
                               verbose=False, thread_count=-1, **params(t))
        m.fit(Xa, ya, eval_set=(Xv, yv), use_best_model=True)
        t.set_user_attr("n", m.get_best_iteration() + 1)
        return m.get_best_score()["validation"]["AUC"]

    b = tune(obj, n_trials, seed, timeout)
    p = b.params
    return CatBoostClassifier(iterations=b.user_attrs["n"], random_seed=seed, verbose=False, thread_count=-1,
                              learning_rate=p["lr"], depth=p["depth"], l2_leaf_reg=p["l2"], random_strength=p["rs"],
                              bagging_temperature=p["bt"]).fit(X, y)


def fit_xgb_default(X, y, seed, cat):
    """Library-default XGBoost, the baseline used by the workshop paper."""
    import xgboost as xgb
    return xgb.XGBClassifier(random_state=seed, n_jobs=-1).fit(X, y)


def fit_ebm(X, y, seed, cat):
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(random_state=seed, n_jobs=-1).fit(X, y)


def fit_lr(X, y, seed, cat):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = [i for i in range(X.shape[1]) if i not in cat]
    pre = ColumnTransformer([
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value=-1),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30)), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num)])
    return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=3000)).fit(X, y)


# ---------------------------------------------------------------- GPU models (in-context learners)
def fit_tabicl(X, y, seed, cat):
    from tabicl import TabICLClassifier
    return TabICLClassifier(device="cuda", random_state=seed).fit(X, y)


def fit_tabicl_cache(X, y, seed, cat):
    """TabICL with its context cache enabled, if this tabicl version exposes one."""
    import inspect
    from tabicl import TabICLClassifier
    params = inspect.signature(TabICLClassifier.__init__).parameters
    log("   TabICLClassifier params:", {k: str(v.default) for k, v in params.items() if k != "self"})
    key = next((p for p in params if "cache" in p.lower()), None)
    if key is None:
        raise RuntimeError("this tabicl version has no cache option")
    default = params[key].default
    value = True if isinstance(default, bool) or default is None else default
    log(f"   using {key}={value!r}")
    return TabICLClassifier(device="cuda", random_state=seed, **{key: value}).fit(X, y)


def tabpfn(version, **kw):
    def fit(X, y, seed, cat):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        mv = getattr(ModelVersion, version)
        try:
            m = TabPFNClassifier.create_default_for_version(mv, device="cuda", random_state=seed, ignore_pretraining_limits=True, **kw)
        except TypeError:
            m = TabPFNClassifier.create_default_for_version(mv)
            m.set_params(device="cuda", random_state=seed, **kw)
        return m.fit(X, y)
    return fit


def with_context(fit, ratio):
    """Context-construction ablation: keep every churner and sample `ratio` non-churners per churner."""
    def f(X, y, seed, cat):
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        if len(neg) <= ratio * len(pos):
            raise RuntimeError(f"natural ratio already <= 1:{ratio}; skipped")
        neg = np.random.default_rng(seed).choice(neg, int(ratio * len(pos)), replace=False)
        keep = np.sort(np.concatenate([pos, neg]))
        return fit(X[keep], y[keep], seed, cat)
    return f


MODELS = {
    "cpu": {"lr": fit_lr, "lgbm": fit_lgbm, "xgb": fit_xgb, "catboost": fit_catboost, "ebm": fit_ebm},
    "gpu": {"tabicl_v2": fit_tabicl, "tabpfn_3_5": tabpfn("V3_5"), "tabpfn_3_5_fast": tabpfn("V3_5_FAST"),
            "tabpfn_2_5": tabpfn("V2_5")},
    "strong": {"lgbm_200": partial(fit_lgbm, n_trials=200, timeout=3600),
               "catboost_200": partial(fit_catboost, n_trials=200, timeout=3600)},
    "context": {f"{name}__ctx_1to{r}": with_context(fit, r)
                for name, fit in (("tabpfn_3_5", tabpfn("V3_5")), ("tabicl_v2", fit_tabicl)) for r in (1, 2)},
}.get(GROUP, {})
if BASE == "ecommerce" and GROUP == "cpu":
    MODELS["xgb_default"] = fit_xgb_default


def predict(m, X):
    """One call when it fits; on GPU OOM fall back to the largest chunk size that fits (each chunk re-encodes the context)."""
    for chunk in (len(X), 10000, 2000):
        try:
            return np.concatenate([m.predict_proba(X[i:i + chunk])[:, 1] for i in range(0, len(X), chunk)])
        except Exception as e:
            if chunk == 2000:
                raise
            log(f"   predict with chunk={chunk} failed ({type(e).__name__}), retrying smaller")
            _gpu_reset()


def _gpu_reset():
    if GPU:
        import torch
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def _gpu_peak():
    if not GPU:
        return 0.0
    import torch
    return torch.cuda.max_memory_allocated() / 1e9


# ---------------------------------------------------------------- scoring-throughput benchmark
TIMING_DATASETS = ALL_DATASETS
TIMING_MODELS = None  # None = all


def timing(sizes=(1000, 10000, 50000), budget_s=1500):
    """Seed-0 fit per dataset, then time scoring of n rows resampled from the test split (after a warm-up call)."""
    import lightgbm as lgb
    import torch
    models = {"lgbm_1000trees": lambda X, y, s, c: lgb.LGBMClassifier(n_estimators=1000, num_leaves=64, verbose=-1).fit(X, y),
              "tabicl_v2": fit_tabicl, "tabicl_v2_cache": fit_tabicl_cache, "tabpfn_3_5": tabpfn("V3_5"),
              "tabpfn_3_5_fast": tabpfn("V3_5_FAST"), "tabpfn_3_5_cache": tabpfn("V3_5", fit_mode="fit_with_cache"),
              "tabpfn_2_5": tabpfn("V2_5")}
    models = {k: v for k, v in models.items() if TIMING_MODELS is None or k in TIMING_MODELS}
    rng = np.random.default_rng(0)
    for ds in TIMING_DATASETS:
        X, y, cat, cols, side = globals()[f"load_{ds}"]()
        log(f"== {ds}: X={X.shape} duplicate rows removed={side['n_dup']}")
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=0)
        for name, fit in models.items():
            try:
                _gpu_reset()
                t0 = time.time(); m = fit(X[itr], y[itr], 0, cat); fit_s = time.time() - t0
                predict(m, X[ite[:200]])  # warm-up
                per10k = None
                for n in sizes:
                    if per10k is not None and per10k * n / 1e4 > budget_s:
                        log(f"   {name} n={n} skipped (est {per10k * n / 1e4:.0f}s)"); continue
                    Xs = X[ite][rng.integers(0, len(ite), n)]
                    torch.cuda.synchronize(); t0 = time.time()
                    predict(m, Xs)
                    torch.cuda.synchronize(); s = time.time() - t0
                    per10k = s / n * 1e4
                    RUNS.append(dict(dataset=ds, model=name, n_scored=n, n_context=len(itr), fit_s=fit_s, score_s=s,
                                     s_per_10k=per10k, gpu_peak_gb=_gpu_peak(), n_dup_removed=side["n_dup"]))
                    log(f"   {name} n={n} score={s:.1f}s ({per10k:.2f}s/10k) gpu={_gpu_peak():.1f}GB")
            except Exception as e:
                log(f"   {name} FAILED {type(e).__name__}: {str(e)[:400]}")
                RUNS.append(dict(dataset=ds, model=name, status="failed", error=f"{type(e).__name__}: {e}"[:1000]))
            json.dump(RUNS, open(f"{OUT}/timing.json", "w"), indent=1)


# ---------------------------------------------------------------- run
def main():
    if GPU:
        import torch
        from tabpfn.constants import ModelVersion
        log("GPU", torch.cuda.get_device_name(0), "| TABPFN_TOKEN loaded:", bool(os.environ.get("TABPFN_TOKEN")))
        log("ModelVersion members:", [v.name for v in ModelVersion])
    from importlib.metadata import version
    for p in ("tabpfn", "tabicl", "lightgbm", "xgboost", "catboost", "interpret", "optuna", "scikit-learn"):
        try:
            log(p, version(p))
        except Exception:
            pass
    if GROUP == "timing":
        return timing()

    X, y, cat, cols, side = globals()[f"load_{BASE}"]()
    dstats = {k: side[k] for k in ("n_dup", "n_conflict_groups", "n_conflict_rows", "churn_rate_before",
                                   "churn_rate_after_dedup")}
    log(f"== {DATASET}: X={X.shape} churn={y.mean():.4f} cat={len(cat)} duplicates={dstats} dedup={DEDUP}")
    np.savez_compressed(f"{OUT}/data_side.npz", y=y, tenure=side["tenure"] if side["tenure"] is not None else np.array([]),
                        revenue=side["revenue"] if side["revenue"] is not None else np.array([]))
    json.dump({"dataset": DATASET, "n": len(y), "n_feat": X.shape[1], "churn_rate": float(y.mean()), "cols": cols,
               "cat_idx": cat, "n_dup_removed": side["n_dup"] if DEDUP else 0, "duplicates": dstats},
              open(f"{OUT}/data_meta.json", "w"), indent=1)

    idx = np.arange(len(y))
    for seed in SEEDS:
        if DEDUP:
            itr, itmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
            ical, ite = train_test_split(itmp, test_size=0.5, stratify=y[itmp], random_state=seed)
        else:  # split whole duplicate groups, stratified on whether the group contains a churner
            ug, inv = np.unique(side["group"], return_inverse=True)
            glab = np.zeros(len(ug), dtype=int); np.maximum.at(glab, inv, y)
            gtr, gtmp = train_test_split(np.arange(len(ug)), test_size=0.4, stratify=glab, random_state=seed)
            gcal, gte = train_test_split(gtmp, test_size=0.5, stratify=glab[gtmp], random_state=seed)
            itr, ical, ite = (np.where(np.isin(inv, g))[0] for g in (gtr, gcal, gte))
        for name, fit in MODELS.items():
            run = dict(dataset=DATASET, model=name, seed=seed, n_train=len(itr), n_test=len(ite))
            try:
                _gpu_reset()
                t0 = time.time(); m = fit(X[itr], y[itr], seed, cat); run["fit_s"] = time.time() - t0
                t0 = time.time(); pte = predict(m, X[ite]); run["pred_s_per_10k"] = (time.time() - t0) / len(ite) * 1e4
                pcal = predict(m, X[ical])
                run.update(status="ok", gpu_peak_gb=_gpu_peak())
                np.savez_compressed(f"{OUT}/preds/{name}_s{seed}.npz", idx_train=itr, idx_cal=ical, idx_test=ite,
                                    p_cal=pcal, p_test=pte)
                from sklearn.metrics import roc_auc_score
                log(f"   {name} s{seed} fit={run['fit_s']:.0f}s pred/10k={run['pred_s_per_10k']:.1f}s "
                    f"gpu={run['gpu_peak_gb']:.1f}GB auc={roc_auc_score(y[ite], pte):.4f}")
            except Exception as e:
                run.update(status="failed", error=f"{type(e).__name__}: {e}"[:2000])
                log(f"   {name} s{seed} FAILED", run["error"][:600])
                if seed == SEEDS[0]:
                    traceback.print_exc()
            RUNS.append(run)
            json.dump(RUNS, open(f"{OUT}/runs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
