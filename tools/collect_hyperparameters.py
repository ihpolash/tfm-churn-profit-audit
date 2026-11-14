"""Collect the selected GBDT configurations recorded by the `retune` kernels into one machine-readable file.

The original prediction runs did not log the configuration Optuna selected. The `retune` kernels rerun the same tuning
with the same seeds and log it, and compare their predictions with the released ones; a configuration is marked as
reproducing the released run only when the predictions agree. Also records the configurations of the GBDTs tuned on
EMP (*_emp), whose predictions are part of the release.

Standard library only:  python3 tools/collect_hyperparameters.py
Writes results/gbdt_hyperparameters.csv (one row per dataset, model, tuning objective and split).
"""
import csv, glob, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-6  # largest absolute probability difference still counted as reproducing the released prediction

rows = []
for f in sorted(glob.glob(str(ROOT / "results/kernels/*-retune*/runs.json"))):
    for r in json.load(open(f)):
        if r.get("status") != "ok" or "params" not in r:
            continue
        name = r["model"]
        base, objective = (name[:-6], "auc") if name.endswith("_rerun") else (name[:-4], "emp")
        diff = r.get("max_abs_diff_test")
        rows.append(dict(dataset=r["dataset"], model=base, tuning_objective=objective, seed=r["seed"],
                         params=json.dumps(r["params"], sort_keys=True), n_trees=r["n_trees"],
                         n_trials_run=r["n_trials_run"], inner_validation_score=r["best_value"],
                         max_abs_diff_vs_released=diff if objective == "auc" else "",
                         reproduces_released=("" if objective == "emp" else
                                              str(diff is not None and diff <= TOL and bool(r.get("same_split"))))))
rows.sort(key=lambda r: (r["dataset"], r["model"], r["tuning_objective"], r["seed"]))
out = ROOT / "results/gbdt_hyperparameters.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
auc = [r for r in rows if r["tuning_objective"] == "auc"]
print(f"wrote {out.relative_to(ROOT)}: {len(rows)} configurations; "
      f"{sum(r['reproduces_released'] == 'True' for r in auc)}/{len(auc)} AUC-tuned reruns reproduce the released predictions")
