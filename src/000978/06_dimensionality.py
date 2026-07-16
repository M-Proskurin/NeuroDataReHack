"""Stage 5 — intrinsic dimensionality (000978), triangulated.

Same estimators as the 000447 stage 5 (src/common/dimensionality.py), on the
smoothed 50 ms **run-epoch** data (sigma=100 ms, speed > 4 cm/s), per file /
region. The decoded task label here is the **run session** (1..N): decode_label
asks how many latent dimensions are needed to tell sessions apart, i.e. how many
dimensions carry the within-day learning drift. ZT2 is excluded (two separate
days) by the loader.

Outputs:
    data/processed/000978/stage5_dim_summary.csv
    data/processed/000978/stage5_dim_curves.csv

Usage:
    pixi run python src/000978/06_dimensionality.py
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))

import argparse

import numpy as np
import pandas as pd

import dimensionality as dim
from bin_smoothing_sensitivity import _smoothed_masked, load_awake_978
from config import RANDOM_SEED, processed_path

DANDISET = "000978"
SIGMA_MS = 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-dim", type=int, default=15)
    args = ap.parse_args()
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"stage 5 dimensionality {DANDISET} (50 ms, sigma={SIGMA_MS} ms, run epochs)")
    data = load_awake_978()                         # streams run epochs, excludes ZT2
    summ, curves = [], []
    for e in data:
        X, pos, sess = _smoothed_masked(e, SIGMA_MS)
        if X.shape[0] < 200:
            continue
        s, c = dim.analyze_group(X, pos, sess, "session", max_dim=args.max_dim, rng=rng)
        if s is None:
            continue
        s.update(session_key=e["key"], region=e["region"])
        summ.append(s)
        for row in c:
            row.update(session_key=e["key"], region=e["region"])
            curves.append(row)
        print(f"  {e['key']:8s} {e['region']}: TwoNN {s['twonn']:.1f}±{s['twonn_sd']:.1f}  "
              f"PR {s['participation_ratio']:.1f}  isomap-knee {s['isomap_knee']}  "
              f"pos-sat {s['decode_pos_sat']} (R²={s['decode_pos_max']:.2f})  "
              f"session-sat {s['decode_label_sat']}", flush=True)

    sdf = pd.DataFrame(summ)
    sdf.to_csv(processed_path("stage5_dim_summary.csv", DANDISET), index=False)
    pd.DataFrame(curves).to_csv(processed_path("stage5_dim_curves.csv", DANDISET), index=False)
    print("\nmedian across files:")
    print(sdf.groupby("region")[["twonn", "participation_ratio", "isomap_knee",
                                 "decode_pos_sat"]].median().round(1).to_string())
    print(f"-> stage5_dim_summary.csv ({len(sdf)} rows), stage5_dim_curves.csv")


if __name__ == "__main__":
    main()
