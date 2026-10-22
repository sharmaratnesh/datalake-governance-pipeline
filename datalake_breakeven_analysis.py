# -*- coding: utf-8 -*-
"""
Step 8 — Break-even analysis: corpus size at which each automated method's total cost falls
below manual annotation, per platform.

Cost model (time-based, no dollar rate given in the brief): for a corpus of N records,
    cost(N) = fixed_overhead_sec + N * per_record_sec
Manual: fixed_overhead=0, per_record = 2100s / 55 records (the brief's human annotation rate).
Automated (laptop-cpu): fixed_overhead=0 — the pipelines have no meaningful one-time setup
    cost beyond a few seconds of model loading, which is already folded into the first
    record's processing_sec in the throughput data.
Automated (hpc-gpu): fixed_overhead = the OBSERVED PBS queue wait time for that job (logged
    explicitly below) — this is the real, often-dominant cost of a shared-cluster deployment
    that a pure per-record-latency number hides. Without it, GPU numbers look unconditionally
    better than they would in practice for small/one-off jobs.

breakeven_N = ceil(fixed_overhead_sec / (manual_rate - method_rate)) when method_rate < manual_rate,
else None (the method never breaks even — its total cost is always >= manual's for any N,
or growing faster, which should not happen for a per-record rate below manual's, but the
check is explicit here rather than assumed).

Emits:
  - evaluation/output/breakeven_<ts>.csv
  - evaluation/output/breakeven_<ts>.png  (cost vs corpus size, one line per method/platform)

Usage:
    python datalake_breakeven_analysis.py
"""
import argparse
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from datalake_artefacts import select_artefact

BASE_DIR = Path(__file__).parent
MANUAL_TOTAL_SEC = 2100.0
MANUAL_N = 55
MANUAL_RATE = MANUAL_TOTAL_SEC / MANUAL_N  # ~38.18 s/record

# Observed PBS queue wait for the successful anchor job (183871.hpc-head01): qtime
# 2026-10-22 00:23:05 -> stime 2026-10-22 01:27:58 = 3893s. This is a real, representative
# wait on this shared small_gpuq queue, not a guess — 0 would understate GPU deployment cost.
HPC_QUEUE_WAIT_SEC = 3893.0


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("breakeven")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def breakeven_n(fixed_overhead_sec: float, method_rate: float):
    if method_rate >= MANUAL_RATE:
        return None
    if fixed_overhead_sec <= 0:
        return 1
    return math.ceil(fixed_overhead_sec / (MANUAL_RATE - method_rate))


def main():
    ap = argparse.ArgumentParser(description="Break-even analysis from a throughput CSV")
    ap.add_argument("--throughput-csv", help="evaluation/output/throughput_<ts>.csv (required if more than one exists)")
    args = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASE_DIR / "evaluation" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / f"log_breakeven_{ts}.txt")

    logger.info("=" * 70)
    logger.info("BREAK-EVEN ANALYSIS (Step 8)")
    logger.info(f"Manual rate: {MANUAL_RATE:.2f} s/record ({MANUAL_TOTAL_SEC}s / {MANUAL_N} records)")
    logger.info("=" * 70)

    throughput_csv = select_artefact(out_dir, "throughput_*.csv", explicit=args.throughput_csv,
                                     what="throughput", option="--throughput-csv")
    if not throughput_csv:
        logger.error("No evaluation/output/throughput_*.csv found — run datalake_tab_throughput.py first. Aborting.")
        return
    tdf = pd.read_csv(throughput_csv)
    logger.info(f"Using throughput file: {throughput_csv.name}")

    rows = []
    for _, r in tdf.iterrows():
        method, platform = r["method"], r["platform"]
        rate = r["mean_latency_sec"]
        if platform == "laptop-cpu":
            overhead = 0.0
        elif str(platform).startswith("hpc-gpu"):
            if HPC_QUEUE_WAIT_SEC is None:
                logger.warning(f"{method} ({platform}): HPC_QUEUE_WAIT_SEC not set — using 0, "
                                "which UNDERSTATES real cluster deployment cost. Set it from the "
                                "job's actual qsub-to-start wait before trusting this row.")
                overhead = 0.0
            else:
                overhead = HPC_QUEUE_WAIT_SEC
        else:
            overhead = 0.0
        be = breakeven_n(overhead, rate)
        rows.append({
            "method": method, "platform": platform, "per_record_rate_sec": rate,
            "fixed_overhead_sec": overhead, "manual_rate_sec": round(MANUAL_RATE, 2),
            "breakeven_n_records": be,
        })
        logger.info(f"{method:<16} {platform:<22} rate={rate:.2f}s/rec  overhead={overhead:.0f}s  "
                    f"breakeven_n={be}")

    out_csv = out_dir / f"breakeven_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info(f"CSV -> {out_csv}")

    # ---- plot: cost vs corpus size ----
    n_max = 3000
    xs = list(range(0, n_max + 1, max(1, n_max // 200)))
    plt.figure(figsize=(9, 6))
    plt.plot(xs, [MANUAL_RATE * n for n in xs], label="Manual", linewidth=2, color="black")
    for row in rows:
        ys = [row["fixed_overhead_sec"] + row["per_record_rate_sec"] * n for n in xs]
        plt.plot(xs, ys, label=f"{row['method']} ({row['platform']})")
    plt.xlabel("Corpus size (records)")
    plt.ylabel("Total cost (seconds)")
    plt.title("Break-even: automated method cost vs. manual annotation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_png = out_dir / f"breakeven_{ts}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    logger.info(f"Plot -> {out_png}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
