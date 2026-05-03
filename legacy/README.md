# legacy/

Archived code paths that are no longer on the active development line. Kept
runnable for reference and reproducibility, but **unsupported** -- the active
codebase under `grassmann/`, `scripts/`, and `tests/` does not import from
here, and changes to shared `grassmann/` modules may break legacy code without
notice.

## legacy/multi_camera/

The N3DV multi-camera training stack (21 simultaneous static cameras x T
frames per scene). It was the active path for Phases 1-7 of the Grassmann
splatting pipeline, but ran into a structural issue documented in
`docs/issues/rca_streak_collapse.md`: the rank-2 Σ_3D parameterization is
geometrically a worldline (a moving line through time), not a 3D ellipsoid.
Attempting to render it from many cameras simultaneously produces edge-on
streaks from any camera other than the one that initialized the splat.

The math spec at `docs/maths/grassmanian_gradients.md` (§5, Theorem
`thm:comparison`) describes the model as designed for **monocular video**
(one camera per frame, possibly moving). The active codebase pivoted there;
the N3DV stack lives here as historical reference.

To reproduce a legacy run:

    modal volume create gs-n3dv
    modal volume put gs-n3dv ./data/n3dv/<scene> /<scene>
    python legacy/multi_camera/scripts/train_n3dv.py train \
        --scene_dir data/n3dv/<scene> \
        --num_iters 30000

(There is no separate Modal entry for the legacy path; the active
`scripts/train_modal.py` targets the monocular pivot. Use a one-off
`modal run` against a custom file if you need to drive it from Modal.)
