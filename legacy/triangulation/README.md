# Legacy: multi-view DLT triangulation

Reference-only archive. `triangulation.py` implements multi-view
DLT triangulation (Hartley & Zisserman, Chapter 12), plus the
reprojection-error and synthetic-observation helpers it depended on.

The live training datasets (NeRFies / HyperNeRF / DyCheck) ship a
COLMAP point cloud per scene, so the production pipeline never
triangulates points itself. The synthetic multi-camera tests + viz
scripts that did call these helpers have themselves been archived
(see `legacy/synthetic/`).

Imports inside the file (`from grassmann.projection import Camera,
project_static`) still point at the live module, so the file can be
imported individually for reference; it just has no caller on the
training path.
