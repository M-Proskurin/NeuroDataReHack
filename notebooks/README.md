# Notebooks

Notebooks are namespaced by dandiset, mirroring `src/`:

- [`000447/`](000447/) — Novel-familiar-novel W-track (CA1–PFC)
- [`000978/`](000978/) — Single-day W-track learning (CA1–PFC)

The two dandisets are analyzed **in parallel, never merged**. Each folder has
one notebook per pipeline stage that mirrors the matching script under
`src/<dandiset>/`; shared utility code lives in `src/common/`. Use notebooks for
exploration and figures; promote stable logic into the scripts so stages stay
independently rerunnable.

```bash
pixi run lab        # launch Jupyter Lab, already activated in the pixi env
```

`read_basics_and_stream.ipynb` (top level) is the upstream DANDI/pynwb streaming
tutorial, kept for reference.

## Running notebooks in VS Code

VS Code launches kernels outside pixi's activation, so a kernel pointed straight
at `.pixi/envs/default/python.exe` **crashes on import** (OpenMP/MKL runtime
conflict). Register a kernel that launches through `pixi run` instead:

```powershell
# register a named kernelspec (once per machine)
pixi run python -m ipykernel install --user `
  --name neurodatarehack --display-name "Python (NeuroDataReHack pixi)"
```

Then edit the generated `kernel.json` (typically
`%APPDATA%\jupyter\kernels\neurodatarehack\kernel.json`) so it launches through
pixi — replace the `argv` with:

```json
"argv": [
  "<path-to>/.pixi/bin/pixi.exe", "run",
  "--manifest-path", "<path-to-repo>/pixi.toml",
  "python", "-Xfrozen_modules=off", "-m", "ipykernel_launcher",
  "-f", "{connection_file}"
]
```

In VS Code: reload the window, then pick **"Jupyter Kernel..." →
"Python (NeuroDataReHack pixi)"**. Launching via `pixi run lab` avoids this
entirely, since the environment is already activated.

The included [.vscode/settings.json](../.vscode/settings.json) points the Python
interpreter at the pixi env and adds `src/common` to the analysis path.
