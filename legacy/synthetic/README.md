# Legacy: synthetic multi-camera scene

Reference-only archive. Files here build and visualize the small
synthetic multi-camera scene that was used to validate early phases
of the implementation (phases 4-6 of the project plan).

The synthetic scene is no longer used by any live test or by the
training entrypoints, which operate on real NeRFies / HyperNeRF /
DyCheck data. Imports in this folder still point at
`grassmann.synthetic`, which has been moved into this directory —
they are kept verbatim and not rewritten for `legacy/` location.
