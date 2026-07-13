# Notebooks

One notebook per pipeline stage, mirroring `src/`. Use notebooks for
exploration and figures; promote stable logic into the matching `src/` script
so stages stay independently rerunnable.

| Notebook | Mirrors | Purpose |
|----------|---------|---------|
| `00_data_sanity_checks.ipynb` | `src/00_download.py`         | Stream one session, QC spikes/epochs/behavior |
| `01_extraction.ipynb`        | `src/01_extraction.py`        | Inspect NWB, sanity-check binned rates |
| `02_baseline_linear.ipynb`   | `src/02_baseline_linear.py`   | PCA / GPFA / dPCA figures |
| `03_nonlinear_embedding.ipynb` | `src/03_nonlinear_embedding.py` | CEBRA / UMAP / Isomap embeddings |
| `04_cross_condition.ipynb`   | `src/04_cross_condition.py`   | Procrustes / CCA comparisons |
| `05_topology.ipynb`          | `src/05_topology.py`          | Persistence diagrams (ripser) |
| `06_dimensionality.ipynb`    | `src/06_dimensionality.py`    | CV reconstruction-error curves |

```bash
pixi run lab        # launch Jupyter Lab, already activated in the pixi env
```

One notebook per pipeline stage under `notebooks/` (see
[notebooks/README.md](notebooks/README.md)). Start with
`notebooks/00_data_sanity_checks.ipynb` for a streamed QC pass on one session.

### Running notebooks in VS Code

VS Code launches kernels outside pixi's activation, so a kernel pointed straight
at `.pixi/envs/default/python.exe` **crashes on import** (OpenMP/MKL runtime
conflict). Register a kernel that launches through `pixi run` instead:

```powershell
# 1. register a named kernelspec (once per machine)
pixi run python -m ipykernel install --user `
  --name neurodatarehack --display-name "Python (NeuroDataReHack pixi)"
```

Then edit the generated `kernel.json` (path printed by the command above,
typically `%APPDATA%\jupyter\kernels\neurodatarehack\kernel.json`) so it launches
through pixi — replace the `argv` with:

```json
"argv": [
  "<path-to>/.pixi/bin/pixi.exe", "run",
  "--manifest-path", "<path-to-repo>/pixi.toml",
  "python", "-Xfrozen_modules=off", "-m", "ipykernel_launcher",
  "-f", "{connection_file}"
]
```

In VS Code: reload the window, then pick **"Jupyter Kernel..." →
"Python (NeuroDataReHack pixi)"**. (Launching via `pixi run lab` instead avoids
this entirely, since the environment is already activated.)

The included [.vscode/settings.json](.vscode/settings.json) points the Python
interpreter at the pixi env and adds `src/` to the analysis path.
