"""Stage 5 (optional) — persistent homology to confirm ring/toroidal structure.

Use `ripser` on the embeddings (or subsampled point clouds) and inspect the
H1/H2 persistence diagrams for the loops expected from W-track navigation.

    from ripser import ripser
    dgms = ripser(points, maxdim=2)["dgms"]

NOTE: giotto-tda is not installed (no py3.11/win-64 wheel). ripser covers the
persistence computation we need here.

TODO: implement after stage 3.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
