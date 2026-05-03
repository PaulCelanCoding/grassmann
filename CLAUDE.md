# Grassmann GS

Grassmannian Gaussian Splatting 

We are implementing docs/maths/. This works with monocular video data.

## Layout
- `grassmann/` — library (Phases 1-7) + `datasets/` (NeRFies, DyCheck loaders)
- `tests/` — pytest (~144 tests)
- `scripts/` — executables (run from any cwd; sys.path nudge included)
- `viz/` — plot generators → `docs/images/`
- `docs/images/` — generated PNGs
- `data/{nerfies,dycheck}/` — datasets (gitignored)
- `legacy/multi_camera/` — archived N3DV/multi-cam stack (unsupported)

## GPU training (Modal, L4)
`modal run scripts/train_modal.py --cmd smoke --dataset nerfies --scene <scene>` (or `train|sanity`); preload data via `modal volume put gs-mono ./data/nerfies/<scene> /<scene>`.

## THE FOLLOWING RULES ARE NON NEGIOTABLE AND ABSOLUTELY MISSION CRITICAL
- Keine `Enhanced`/`Advanced`-Varianten — Legacy-Klassen direkt updaten
- NEVER FAKE RESULTS, ITS BETTER FOR THINGS TO FAIL FAST INSTEAD OF FAKE SUCESS !!! Every Fallback must be documented and throw a dedicated warning. 
- do NOT glaze me; do NOT and NEVER be sycophantic - be objective and nüchtern
- when in plan mode or when i ask for discussion: always ask me as many non trivial questions as possible with the askuser tool! Das betrifft (if applicable) Semantischer Processing Flow, Architektur, Datenfluss, Main entry points, Data structures, Testing, weitere affektierte Bereiche der Code Basis usw usf. Der Vorgang ist folgender:
JEDE Fragerunde hat als letzte Multiple Choice Frage: "Weitere Fragen?" mit möglichen Aspekten die näher befragt werden und "Keine weiteren Fragen" zur Auswahl!!  
- ALWAYS reply in english!
- BE VERY BRIEF
