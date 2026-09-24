"""
Extend every comparison from 6 to 20 seeds, restore the order-selected per-engine predictions
that were not archived for the original 6 seeds, and add the capacity controls for all seeds.

Run from the repository root AFTER the feature caches exist (results/_feature_cache), e.g.
    python3 experiments/run_extra_seeds.py --workers 4

Resumable: a stage is skipped when its output already exists.

Extra seeds were fixed in advance as 1..15 excluding 7 (already used): they were not chosen after
looking at any result.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
OLD_SEEDS = [42, 123, 2024, 7, 99, 314]
NEW_SEEDS = [s for s in range(1, 16) if s != 7]          # 14 new seeds -> 20 total

STAGES = {   # name -> (script, output that marks completion)
    "main":           ("run_dataset.py",                 "results/{fd}/seed_{s}/raw_predictions.json"),
    "order_selected": ("run_dataset_order_selected.py",  "results/order_selected/{fd}/seed_{s}/raw_predictions.json"),
    "causal":         ("run_dataset_causal.py",          "results/causal/{fd}/seed_{s}/raw_predictions.json"),
    "trend_controls": ("run_trend_controls.py",          "results/trend_controls/{fd}/seed_{s}/raw_predictions.json"),
    "capacity":       ("run_capacity_controls.py",       "results/capacity_controls/{fd}/seed_{s}/raw_predictions.json"),
    "cnn":            ("run_cnn_lstm_baseline.py",       "results/cnn_lstm/{fd}/seed_{s}/raw_predictions.json"),
}
os.makedirs("results/logs", exist_ok=True)


def stored_rmse(fd, s):
    p = f"results/order_selected/{fd}/seed_{s}/summary_metrics.json"
    if not os.path.exists(p): return None
    d = json.load(open(p)); return {k: v["RMSE"] for k, v in d.items() if k != "_meta"}


def run_job(job):
    fd, s, stages = job
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for st in stages:
        script, out = STAGES[st]
        if os.path.exists(out.format(fd=fd, s=s)): continue
        before = stored_rmse(fd, s) if (st == "order_selected" and s in OLD_SEEDS) else None
        t0 = time.time()
        with open(f"results/logs/{st}_{fd}_{s}.log", "w") as lf:
            r = subprocess.run([sys.executable, f"experiments/{script}", fd, str(s)], stdout=lf, stderr=subprocess.STDOUT, env=env)
        ok = r.returncode == 0 and os.path.exists(out.format(fd=fd, s=s))
        note = ""
        if ok and before:
            after = stored_rmse(fd, s)
            diff = max(abs(after[k] - before[k]) for k in before)
            note = f" (max |RMSE change| vs stored: {diff:.2e})"
            json.dump({"before": before, "after": after, "max_abs_diff": diff},
                      open(f"results/logs/restore_check_{fd}_{s}.json", "w"))
        print(f"[{time.strftime('%H:%M:%S')}] {fd} seed {s} {st}: {'ok' if ok else 'FAILED'} in {time.time()-t0:.0f}s{note}", flush=True)
        if not ok: return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--datasets", nargs="*", default=DATASETS); ap.add_argument("--only", nargs="*", default=None, help="run only these stages, on the new seeds"); a = ap.parse_args()
    if a.only:
        jobs = [(fd, s, a.only) for s in NEW_SEEDS for fd in a.datasets]
    else:
        jobs = [(fd, s, ["main", "order_selected", "causal", "trend_controls", "capacity"]) for s in NEW_SEEDS for fd in a.datasets]
        jobs += [(fd, s, ["order_selected", "capacity"]) for s in OLD_SEEDS for fd in a.datasets]
    print(f"{len(jobs)} (dataset, seed) jobs, {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_job, jobs))
    print(f"finished: {sum(res)}/{len(jobs)} jobs fully ok", flush=True)
