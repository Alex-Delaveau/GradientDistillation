# pixels_convergence.py
# uv run --with wandb --with pandas python pixels_convergence.py
import re
import pandas as pd
import wandb

PROJECT = "alex26delaveau-lyon-2-/pixels"   # ajuster si le sync affiche autre chose
RUN_RE = re.compile(r"_(?P<prior>none|slurpp|ppg)_ipc(?P<ipc>\d+)_s(?P<seed>\d+)$")

api = wandb.Api()
rows = []

for r in api.runs(PROJECT):
    m = RUN_RE.search(r.name)
    if not m:
        continue
    if r.state != "finished":
        print(f"[skip] {r.name} state={r.state}")
        continue

    f1 = (r.history(keys=["_step", "val/top1"], samples=100000, pandas=True)
           .dropna().sort_values("_step").set_index("_step")["val/top1"])
    dr = (r.history(keys=["_step", "physics/T_drift_rmse"], samples=100000, pandas=True)
           .dropna().sort_values("_step").set_index("_step")["physics/T_drift_rmse"])
    if f1.empty:
        print(f"[skip] {r.name} : pas de val/top1")
        continue

    plateau = f1.iloc[-5:].mean()
    hits = f1[f1 >= 0.95 * plateau]
    rows.append({
        "prior": m["prior"], "ipc": int(m["ipc"]), "seed": m["seed"],
        "plateau": plateau,
        "t95": int(hits.index[0]) if len(hits) else None,
        "drift": float(dr.iloc[-1]) if len(dr) else None,
        "last_step": int(f1.index[-1]),
    })

df = pd.DataFrame(rows).sort_values(["ipc", "prior", "seed"])
df.to_csv("pixels_metrics.csv", index=False)

# --- contrôles ---
print("runs récupérés :", len(df))
print(df.groupby(["ipc", "prior"]).size().unstack(fill_value=0))   # doit être 5 partout
print("steps finaux :", sorted(df.last_step.unique()))             # 4990/5000 uniquement

# --- tableau ---
agg = (df.groupby(["ipc", "prior"])
         .agg(t95_med=("t95", "median"), t95_min=("t95", "min"), t95_max=("t95", "max"),
              plateau_mean=("plateau", "mean"), plateau_std=("plateau", "std"),
              drift_mean=("drift", "mean"), drift_std=("drift", "std"))
         .round(3))
print(agg.to_string())

# --- dérive normalisée à none, par IPC ---
for ipc, g in df.groupby("ipc"):
    base = g.loc[g.prior == "none", "drift"].mean()
    ratios = {p: round(sg.drift.mean() / base, 3) for p, sg in g.groupby("prior")}
    print(f"ipc{ipc} dérive/none : {ratios}")