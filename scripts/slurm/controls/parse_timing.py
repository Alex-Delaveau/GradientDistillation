#!/usr/bin/env python3
"""Summarize timing_controls.slurm logs (stdout lines prefixed with an epoch timestamp).

    python scripts/slurm/controls/parse_timing.py ctl_time_*.out [--csv timing.csv]

Phases (seconds), from the log lines printed by the unmodified code:
  setup     : TIMING_BEGIN -> first [PhysicsFormationDataset] line, or -> "No checkpoint
              found" for the baseline (modules, data, backbone, wandb)
  sample    : "building sample init" -> "building prior init"     (medoid init object)
  prior_load: "building prior init"  -> "selecting samples"       (PPG/SLURPP weights)
  select    : "selecting samples"    -> "computing priors"        (medoid selection)
  prior_inf : "computing priors"     -> "priors ready"            (prior inference)
  init_total: TIMING_BEGIN -> "No checkpoint found" (start of the distillation loop)
  distill   : "No checkpoint found"  -> "Distillation Over"
  it/s      : from the checkpoint lines (every 100 steps), whole run and last 1000 steps
"""
import argparse
import csv
import re
import sys

MARKS = {
    "begin": "TIMING_BEGIN",
    "sample": "building sample init",
    "prior_load": "building prior init",
    "select": "selecting samples",
    "prior_inf": "computing priors",
    "prior_done": "priors ready",
    "loop": "No checkpoint found",
    "over": "Distillation Over",
}


def parse(path):
    first, ckpts = {}, []
    cond = None
    for line in open(path, errors="replace"):
        # $EPOCHREALTIME uses the locale radix ("." or ",")
        m = re.match(r"^(\d+)[.,](\d+) (.*)$", line.rstrip("\n"))
        if not m:
            continue
        t, msg = float(f"{m.group(1)}.{m.group(2)}"), m.group(3)
        if cond is None:
            c = re.search(r"cond=(\S+)", msg)
            cond = c.group(1) if c else None
        for k, pat in MARKS.items():
            if pat in msg and k not in first:
                first[k] = t
        if msg.strip() == "Saved!":
            ckpts.append(t)

    def d(a, b):
        return round(first[b] - first[a], 1) if a in first and b in first else None

    row = {"file": path, "cond": cond,
           "setup": d("begin", "sample") if "sample" in first else d("begin", "loop"),
           "sample": d("sample", "prior_load"), "prior_load": d("prior_load", "select"),
           "select": d("select", "prior_inf"), "prior_inf": d("prior_inf", "prior_done"),
           "init_total": d("begin", "loop"), "distill": d("loop", "over"),
           "n_ckpt": len(ckpts), "it_s": None, "it_s_last1000": None}
    # checkpoints at steps 0,100,...,5000 -> 100 steps between consecutive ones
    if len(ckpts) >= 2:
        row["it_s"] = round(100 * (len(ckpts) - 1) / (ckpts[-1] - ckpts[0]), 3)
    if len(ckpts) >= 11:
        row["it_s_last1000"] = round(1000 / (ckpts[-1] - ckpts[-11]), 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = [parse(p) for p in a.logs]
    keys = list(rows[0])
    w = csv.DictWriter(open(a.csv, "w", newline="") if a.csv else sys.stdout, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
