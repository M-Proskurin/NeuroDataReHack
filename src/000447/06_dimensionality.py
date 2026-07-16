"""Stage 5 — intrinsic dimensionality (000447), triangulated.

Per CLAUDE.md we do not trust any single estimator: on the smoothed 50 ms awake
data (sigma=100 ms, speed > 4 cm/s) we report, per subject / region / condition
(novel, familiar, all):

  * TwoNN            — primary model-free intrinsic dimension (bootstrap mean±sd)
  * participation_ratio — linear UPPER bound (curved manifolds inflate it)
  * isomap_knee      — dimension where Isomap residual variance drops below 5%
  * decode_pos_sat   — PCA dims to saturate cross-validated position decoding
  * decode_label_sat — PCA dims to saturate novelty (novel/familiar) decoding

The estimators live in src/common/dimensionality.py; here we just load data and
loop. The PR−TwoNN gap flags curvature (linear over-counting).

Outputs:
    data/processed/000447/stage5_dim_summary.csv   (one row per subject/region/condition)
    data/processed/000447/stage5_dim_curves.csv    (dim-resolved isomap/decoding curves)

Usage:
    pixi run python src/000447/06_dimensionality.py
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse

import numpy as np
import pandas as pd

import dimensionality as dim
from bin_smoothing_sensitivity import _smoothed_masked, load_awake_447
from config import RANDOM_SEED, processed_path

DANDISET = "000447"
SIGMA_MS = 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-dim", type=int, default=15)
    args = ap.parse_args()
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"stage 5 dimensionality {DANDISET} (50 ms, sigma={SIGMA_MS} ms)")
    data = load_awake_447()
    summ, curves = [], []
    for e in data:
        X, pos, cond = _smoothed_masked(e, SIGMA_MS)   # X: (T, N) smoothed, speed-filtered
        subsets = [("all", np.ones(len(cond), bool))]
        subsets += [(c, cond == c) for c in ("novel", "familiar")]
        for name, mask in subsets:
            if mask.sum() < 200:
                continue
            s, c = dim.analyze_group(X[mask], pos[mask], cond[mask], "novelty",
                                     max_dim=args.max_dim, rng=rng)
            if s is None:
                continue
            s.update(subject=e["key"], region=e["region"], condition=name)
            summ.append(s)
            for row in c:
                row.update(subject=e["key"], region=e["region"], condition=name)
                curves.append(row)
            print(f"  {e['key']:14s} {e['region']} {name:8s}: "
                  f"TwoNN {s['twonn']:.1f}±{s['twonn_sd']:.1f}  PR {s['participation_ratio']:.1f}  "
                  f"isomap-knee {s['isomap_knee']}  pos-sat {s['decode_pos_sat']} "
                  f"(R²={s['decode_pos_max']:.2f})  novelty-sat {s['decode_label_sat']}", flush=True)

    sdf = pd.DataFrame(summ)
    sdf.to_csv(processed_path("stage5_dim_summary.csv", DANDISET), index=False)
    pd.DataFrame(curves).to_csv(processed_path("stage5_dim_curves.csv", DANDISET), index=False)

    a = sdf[sdf.condition == "all"]
    print("\nmedian across subjects (condition=all):")
    print(a.groupby("region")[["twonn", "participation_ratio", "isomap_knee",
                               "decode_pos_sat"]].median().round(1).to_string())
    print(f"-> stage5_dim_summary.csv ({len(sdf)} rows), stage5_dim_curves.csv")


if __name__ == "__main__":
    main()
