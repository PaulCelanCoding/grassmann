# Grassmann GS — Repo Map

Grassmannian Gaussian Splatting für Multi-View Video. Phasen 1-7 + N3DV-Training-Stack.

## Layout
- `grassmann/` — Library (Phasen 1-7).
  - Phase 1: `quaternion.py`, `grassmann.py`
  - Phase 2: `projection.py`, `jacobian.py`
  - Phase 3: `gaussian.py`, `rasterizer.py`
  - Phase 4: `synthetic.py`, `triangulation.py`, `initialization.py`
  - Phase 5: `trainable.py`, `losses.py`, `training.py`
  - Phase 6: `density_control.py`
  - Phase 7: `fast_rasterizer.py` (Inria CUDA-Adapter, fällt auf Phase-3-Toy zurück wenn nicht verfügbar)
- `tests/` — pytest, ~113 Tests. `python -m pytest tests/ -v`
- `scripts/` — Executables (vom Repo-Root aufrufen): `train_n3dv.py`, `diagnose_n3dv.py`, `sanity_one_gaussian.py`, `benchmark_phase7.py`, `stress_test_jacobian.py`, `preprocess.sh`, `colmap.sh`
- `viz/` — Plot-Generatoren, schreiben nach `docs/images/`
- `docs/images/` — Generierte Phase-/Demo-PNGs (im Repo eingecheckt)
- `data/n3dv/` — Datasets, gitignored

## Konventionen
- Keine `Enhanced`/`Advanced`-Varianten — Legacy-Klassen direkt updaten, API stabil halten.
- Keine fake fallbacks; lieber fail-fast.
- Scripts laufen mit cwd = Repo-Root (relative Pfade wie `data/n3dv/...`, `docs/images/...`).
