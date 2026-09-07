# %% [markdown]
# ## Convergence and drift
#
# t95 = first step at which val/top1 (logged as macro F1) reaches 95% of the
# plateau, the plateau being the mean of the last five logged points.
# Drift = physics/T_drift_rmse at the final step, normalized to `none`.

# %%
import re
from datetime import datetime

import numpy as np
import pandas as pd
import wandb

ENTITY, PROJECT = "alex26delaveau-lyon-2-", "Linear-Gradient-Matching"
CUTOFF = "2026-08-17T00:00:00"
RUN_RE = re.compile(r"^dinov2_vitb_aqua20_(none|slurpp|ppg|baseline)_ipc(\d+)_s(\d+)$")

F1_KEY    = "val/top1"                 # logged as macro F1 despite the name
DRIFT_KEY = "physics/T_drift_rmse"
PLATEAU_N = 5
THRESHOLD = 0.95

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}",
                filters={"createdAt": {"$gte": CUTOFF}})
print(f"{len(runs)} runs after the date filter")

# %% sanity check on the available keys, on the first matching run
for run in runs:
    if RUN_RE.match(run.name):
        print(f"keys in {run.name}:")
        print(sorted(k for k in run.summary.keys() if not k.startswith("_")))
        break

# %%
rows, skipped = [], []
for run in runs:
    m = RUN_RE.match(run.name)
    if m is None:
        skipped.append((run.name, "name"))
        continue

    prior, ipc, seed = m.group(1), int(m.group(2)), int(m.group(3))

    h = run.history(pandas=True)
    if h.empty or F1_KEY not in h.columns:
        skipped.append((run.name, "no history"))
        continue

    f1 = h[["_step", F1_KEY]].dropna().sort_values("_step")
    if len(f1) < PLATEAU_N + 1:
        skipped.append((run.name, f"only {len(f1)} points"))
        continue

    plateau = f1[F1_KEY].iloc[-PLATEAU_N:].mean()
    reached = f1[f1[F1_KEY] >= THRESHOLD * plateau]
    t95 = int(reached["_step"].iloc[0]) if len(reached) else np.nan

    drift = np.nan
    if DRIFT_KEY in h.columns:
        d = h[["_step", DRIFT_KEY]].dropna().sort_values("_step")
        if len(d):
            drift = float(d[DRIFT_KEY].iloc[-1])

    rows.append({"prior": prior, "ipc": ipc, "seed": seed,
                 "plateau": plateau, "t95": t95, "drift": drift,
                 "n_points": len(f1), "created": run.created_at})

df = pd.DataFrame(rows).sort_values(["ipc", "prior", "seed"])
df.to_csv("convergence_raw.csv", index=False)

print(f"\n{len(df)} runs kept, {len(skipped)} skipped")
for name, why in skipped:
    print(f"  [skip] {name}: {why}")

# %% coverage check: 5 seeds per (ipc, prior)
counts = df.groupby(["ipc", "prior"]).size().unstack(fill_value=0)
print("\nruns per condition (expect 5 everywhere):")
print(counts.to_string())
if (counts.values[counts.values > 0] != 5).any():
    print("!! some conditions do not have exactly 5 seeds")

# duplicate seeds would silently bias the means
dups = df[df.duplicated(["prior", "ipc", "seed"], keep=False)]
if len(dups):
    print("\n!! duplicate (prior, ipc, seed):")
    print(dups[["prior", "ipc", "seed", "created"]].to_string(index=False))

# %%
summary = (df.groupby(["ipc", "prior"])
             .agg(plateau_mean=("plateau", "mean"), plateau_std=("plateau", "std"),
                  t95_mean=("t95", "mean"), t95_std=("t95", "std"),
                  drift_mean=("drift", "mean"), drift_std=("drift", "std"),
                  n=("t95", "count"))
             .reset_index())

ref = summary[summary["prior"] == "none"].set_index("ipc")["drift_mean"]
summary["drift_norm"] = summary.apply(
    lambda r: r.drift_mean / ref.get(r.ipc, np.nan), axis=1)

print("\n" + summary.round(3).to_string(index=False))
summary.to_csv("convergence_summary.csv", index=False)

# %% LaTeX rows, IPC 1
sub = summary[summary["ipc"] == 1].set_index("prior")
print("\n% LaTeX rows (IPC 1)")
for p in ("slurpp", "ppg", "none", "baseline"):
    if p not in sub.index:
        continue
    r = sub.loc[p]
    dn = "--" if np.isnan(r.drift_norm) else f"${r.drift_norm:.2f}$"
    print(f"    \\texttt{{{p}}} & {dn} & ${r.t95_mean:.0f} \\pm {r.t95_std:.0f}$ \\\\")