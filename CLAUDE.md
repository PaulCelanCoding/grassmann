# Grassmann GS

Grassmannian Gaussian Splatting (Phases 1-7) + N3DV training stack.

We are implementing docs/maths/. This works with monocular video data.

## Layout
- `grassmann/` — library (Phases 1-7)
- `tests/` — pytest (~113 tests)
- `scripts/` — executables (run from repo root)
- `viz/` — plot generators → `docs/images/`
- `docs/images/` — generated PNGs
- `data/n3dv/` — datasets (gitignored)

## GPU training (Modal, L4)
`modal run scripts/train_modal.py --cmd smoke` (or `train|diagnose|sanity`); preload data once via `modal volume put gs-n3dv ./data/n3dv/<scene> /<scene>`.

## THE FOLLOWING RULES ARE NON NEGIOTABLE AND ABSOLUTELY MISSION CRITICAL
- Keine `Enhanced`/`Advanced`-Varianten — Legacy-Klassen direkt updaten
- NEVER FAKE RESULTS, ITS BETTER FOR THINGS TO FAIL FAST INSTEAD OF FAKE SUCESS !!! Every Fallback must be documented and throw a dedicated warning. 
- do NOT glaze me; do NOT and NEVER be sycophantic - be objective and nüchtern
- when in plan mode or when i ask for "discussion": always ask me as many non trivial questions as possible with the askuser tool! Das betrifft (if applicable) Semantischer Processing Flow, Architektur, Datenfluss, Main entry points, Data structures, Testing, weitere affektierte Bereiche der Code Basis usw usf. Der Vorgang ist folgender:
JEDE Fragerunde hat als letzte Frage: "Weitere Fragen?" mit möglichen Aspekten die näher befragt werden und "Keine weiteren Fragen" zur Auswahl!!  
- ALWAYS reply in english!
- BE VERY BRIEF
