# Grassmann Splatting — Phasenplan (v3)

Inkrementelle Implementierung mit testbaren Invarianten. Pro Phase werden
zwei Achsen unterschieden:

- **Implementiert:** Code existiert, prüft Mathematik-/API-Konsistenz
- **Validiert:** empirisch verifiziert auf realistischer Skala (Konvergenz,
  Stabilität, korrekte Outputs auf echten Daten)

**Eine Phase ist erst abgeschlossen, wenn beide Achsen grün sind.** Das
unterscheidet sich vom bisherigen Test-Apparat, der primär Achse 1 prüft.

**Aktueller Stand:** P0–P3 vollständig (P4 mit Vorbehalt). Ab P5 driftet die
Validierung — Tests laufen grün, aber das System konvergiert auf echten
Daten nicht. Die Aufgabe der nächsten Iteration ist, die Validierungs-Achse
schrittweise zu schließen.

**Änderungen gegenüber v2:** Loss-Werte in P5 präzisiert; P4 von ✅ auf ⚠
(end-to-end gradcheck fehlt); P3 Bit-Vergleich → numerischer Vergleich; P7
Splitting-Test als Stetigkeit; P8 Adam-Reset mit konkreten Iter-Verlusten;
P10 Hypothese (a) ersetzt durch (p,q)-Perturbation plus strukturelle
Beobachtung zur t-Skalierung; Falsifikations-Test vor P10; Risk-Register;
Test-Form-Faktor; Logging um `cond_sigma_k` und `mean_c` erweitert.

---

## Was die existierenden 113 Tests *nicht* leisten

Tests wie `test_init_gaussian_mean_matches_point_approximately` prüfen, ob
ein Gaussian an einer 3D-Position platziert wird (ja, korrekt). Sie prüfen
**nicht**, ob diese Platzierung als Trainings-Initialisierung sinnvoll ist
(nein — siehe P10). Konsistenz der Mathematik ≠ Tauglichkeit des
Algorithmus. Validierungs-Tests müssen anders aussehen: nicht "tut der Code
was er sagt", sondern "produziert er auf realistischer Skala das gewünschte
Ergebnis".

---

## P0 — Geometrische Primitive

**Implementiert (✅):** Quaternion-Algebra, `line_to_pq`, `pq_to_basis`.
`test_quaternion` (11) + `test_grassmann` (16) inkl. Issue-1- und
Issue-3-Tests.

**Validiert (✅):** Mathematik ist algebraisch — kein Skalen-Sprung möglich.

---

## P1 — `compute_derived` (statisch, t = 1)

**Implementiert (✅):** Map zu (V_3D, Σ_3D). `test_jacobian` deckt FD-
Vergleich ab.

**Validiert (✅):** rein algebraisch wie P0.

---

## P2 — Zeit-Konditionierung

**Implementiert (✅):** Eq (37), (44), (45) inkl. Issue-2-Fix
(`sigma_tt_pure` getrennt von `Σ_tt`). `test_jacobian` (14),
`test_sigma_3d_rank_drops_after_conditioning`.

**Validiert (✅):** algebraisch plus Cross-Check gegen explizite 4D-
Marginalisierung.

---

## P3 — Forward-Render

**Implementiert (✅):** Toy-Rasterizer + Inria-CUDA-Adapter. `test_rendering`
(18), `test_rasterize_temporal_fade`, `test_rasterize_occlusion`.

**Validiert (⚠ teilweise):**
- Toy-Pfad auf Synthetik mit ≤ 100 Splats: ✅
- CUDA-Pfad auf realer GPU: **nicht ausgeführt** (`is_available()` returnt
  False im Test-Setup, alle 10 Tests laufen den Fallback)
- Numerischer Vergleich Toy ↔ CUDA: **fehlt**

**Was fehlt:** GPU-Run mit 1k–10k Splats; numerischer Vergleich Toy ↔ CUDA
mit Toleranz (L1 < 1e-2 bei festem Seed, plus visueller A/B). Kein Bit-
Vergleich erwartbar — Toy macht naive Front-to-Back, Inria macht Tile-
basiertes Sorting; algorithmisch verschieden.

---

## P4 — Autograd-Backward

**Implementiert (✅):** alles pure PyTorch, Autograd. `test_initialization`
(21).

**Validiert (⚠ teilweise):** Was existiert ist
`test_jacobian_full_static_finite_difference` (analytisch vs. FD) und
`test_jacobian_full_static_matches_autograd` (analytisch vs. Autograd). Per
Transitivität validiert das den statischen Jacobi. **Aber:** kein
`torch.autograd.gradcheck` durch die zusammengesetzte Pipeline
`compute_derived ∘ condition_on_time → Render → Loss`. Der End-to-End-Pfad
ist nicht verifiziert.

**Was fehlt:** `torch.autograd.gradcheck` mit double precision auf der
zusammengesetzten Pipeline (~30 min Aufwand, ~10 Splats wegen
Laufzeit). Action-Item.

---

## P5 — Riemannsche Optimierung (Spielzeug)

**Implementiert (✅):** Euclidean Adam + Double-Renormalisierung
(`trainable.py:99-102`, `Trainer.renormalize_manifolds()`).

**Validiert (✅ Toy, ⚠ Annahme nicht stress-getestet):**
- Single-Gaussian-Konvergenz auf Synthetik: ✅
- Multi-View-Multi-Frame, 3 Splats:
  Loss 0.047 (it 20) → 0.022 (it 100) → 0.020 (it 200), monotones Plateau
- **Antidiagonal-Stress-Test fehlt** (Init bei p · q = -0.95)
- **Strategie-Vergleich Euclidean vs. projiziert vs. Riemannsch fehlt** —
  offene Forschungsfrage; Risk-Item bei Showstopper-Risiko (siehe Register)

---

## P6 — Single-Scene Overfit, feste Splat-Anzahl

Hier beginnt die Validierungs-Lücke.

**Implementiert (✅):** `test_trainer_multi_view_multi_frame` läuft.

**Validiert (❌):** der existierende Test ist 3 Splats auf Synthetik. Das
ist kein Skalen-Test. Was fehlt:

- 100 → 1k → 10k Splats, 5–20 Views, 30k Iter
- **Konkrete Erfolgs-Schwellen** relativ zu Standard-3DGS-Baseline auf
  identischer Synthetik:
  - Δ < 3 dB PSNR: passable, validiert die Methode
  - Δ < 1 dB PSNR: SOTA-konkurrenzfähig
  - Δ > 5 dB: Methode hat strukturelles Problem
- Loss-Kurven-Inspektion mit Antidiagonal-Logging (siehe unten)

**Test-Form-Faktor (Entscheidung):** Skalen-Tests werden mit
`@pytest.mark.scale` markiert und mit `pytest --runscale` aktiviert,
default skip in CI. Separates `scale_study.py` als Driver für Multi-Splat-
Konfigurationen mit Loss-Plot-Output. Ohne diese Operationalisierung
wandert P6 nie von ❌ zu ✅.

**Was wir aus dem N3DV-Versuch wissen:** der Sprung von 3 auf 30k Splats
bricht das System. Wir wissen nicht, ob es bei 100, 1000 oder erst bei
30k bricht. Diese Skalen-Studie ist der direkte Diagnose-Pfad.

**Reihenfolge:** P6 bei 100/1k Splats läuft auf CPU mit Toy-Rasterizer in
Minuten — kann parallel zu P3-GPU-Verifikation starten. CUDA wird erst für
10k+ benötigt; das ist der natürliche Sync-Punkt.

---

## P7 — Densification (Mechanik)

**Implementiert (✅):** Clone, Split, Pruning manifold-erhaltend.
`test_density_control` (12).

**Validiert (❌):**
- "Synthetic Recovery" (zu wenige Splats ⇒ Densification rekonstruiert
  Detail) ist nicht als End-to-End-Konvergenz getestet, nur als Mechanik
- **Splitting-Stetigkeit fehlt:** Render-Loss-Differenz vor/nach Split auf
  identischem Frame *ohne* Optimizer-Step muss klein sein. (Identität ist
  mathematisch nicht haltbar — Σ-Shrink um φ=1.6 plus Mean-Offset, plus
  nicht-lineare Alpha-Compositing-Akkumulation. Stetigkeit ist das
  korrekte Kriterium.)
- Wir wissen nicht, ob das Manifold-Splitting in der Praxis Detail
  rekonstruiert oder nur Splats vermehrt.

---

## P8 — Adaptive Density Control im Trainings-Loop

**Implementiert (⚠ partiell):**
- Density-Schedule existiert
- **Adam-Reset bei Density-Operationen** wirft alle Moments weg, auch
  für die unverändert behaltenen Splats. Standard-3DGS erhält Moments für
  bestehende Splats, initialisiert nur neue auf 0
  (`cat_tensors_to_optimizer` + `replace_tensor_to_optimizer`).
  **Konkrete Auswirkung:** bei `densify_every=500`, `densify_stop=15000`
  = 30 Density-Events × ~50–100 Iter Adam-Re-Equilibrierung = **1500–3000
  verlorene Trainings-Iterationen**.
- **Kein Opacity-Reset** implementiert
- **Kein Antidiagonal-Logging** im Loop

**Validiert (❌):** noch nicht angefangen — siehe P10.

---

## P9 — Custom-CUDA-Backward

**Skip.** Wir nutzen Autograd durch `compute_derived`/`condition_on_time`,
nur der Rasterizer-Kernel ist CUDA. Custom-CUDA wäre spätere Performance-
Optimierung.

---

## Falsifikations-Test vor P10 (60-Sekunden-Sprint)

Bevor P10-Diagnose-Hypothesen einzeln durchgespielt werden, ein
disambiguierender Test: **Initial-State von `train_n3dv.py` mit
`σ_αα = 1e-8` rendern, ohne Training.**

- **Wenn die Splats Punkte sind:** σ_αα-Streckung ist die Streichen-Quelle,
  Hypothese 1 (Strichholz-Init) bestätigt — Fix muss an die J_embed-
  Struktur ran.
- **Wenn immer noch Streifen:** liegt nicht an σ_αα-Streckung, sondern an
  z. B. σ_k Pixel-Blur oder Tile-Komposition. Hypothese 2 priorisieren.

Eliminiert in 60 Sekunden die Hälfte der Hypothesen-Liste.

---

## P10 — Echte Daten (N3DV)

**Implementiert (✅):** Pipeline läuft.

**Validiert (❌, blockiert):** sichtbarer Collapse — radiale Streifen vom
Bildzentrum (`diagnose_render_iter100.png`).

### Strukturelle Beobachtung zur t-Skalierung

Die Issue-3-Fix-Skalierung `line_to_pq(c_ref/t, û)` produziert mit
wachsendem t einen schrumpfenden y-Input. Das treibt c = p·q → 1
(Antidiagonale am anderen Ende — die Diagonale, wo p = q). Numerisch
verifiziert mit realistischem c_ref:

| t   | c        | \|e2_spatial\| | Anisotropie-Verhältnis |
|-----|----------|----------------|------------------------|
| 1   | -0.61    | 0.69           | ~1:1                   |
| 50  | 0.9949   | 0.05           | ~20:1                  |
| 300 | 0.9999   | 0.008          | ~430:1                 |

**Konsequenz:** Splats für späte N3DV-Frames sind systematisch noch viel
schlechter konditioniert als Splats für frühe Frames. Das ist kein Bug,
sondern strukturell im φ_t-Embedding mit `c_ref/t`-Skalierung. Bei 300
Frames wird der Effekt akut.

### Hypothesen, in absteigender Wahrscheinlichkeit

1. **Strichholz-Init via Diagonal-Approach (strukturell, kritisch).**
   Bei N3DV-Init ist c ≈ 0.99 (nicht exakt 1), |e2_spatial| ≈ 0.05.
   Σ_3D ist nicht exakt rank-1, aber Anisotropie-Verhältnis 20–430:1
   produziert visuell Strichhölzer.
2. **σ_k API-Doppelrolle (kritisch, Engineering-Bug).**
   Derselbe Parameter ist in `gaussian.py:155` temporale Varianz und in
   `rasterizer.py:80` 2D-Pixel-Varianz. Bei `sigma_k = 20` als Pixel-
   Varianz: Std 4.5 Pixel.
3. **30k zufällige Punkte in 3m-Cube** statt COLMAP-Init.
4. **lr_pq = 1e-3** zu klein für Manifold-Reparatur.

### Diagnose-Sprint (orthogonal, einzeln testen)

- **(a''') (p,q)-Perturbation auf S²:** rotiere (p,q) bei Init um 25–30°
  auf S². Das zwingt c ≤ 0.9 mechanisch und produziert echte rank-2
  J_embed-Spalten. (σ_αβ = 0.001 allein ist zu schwach — Beitrag ist
  Faktor 100 unter σ_αα-Beitrag, Anisotropie bleibt strukturell.)
- **(b) σ_k API-Split:** trennen in `sigma_k_pixel²` (Default 0.3) und
  `sigma_k_temporal²` (Default unverändert).
- **(c) COLMAP-Init:** `points3D.txt` als Punkt-Quelle statt Random-Cube.

**Wichtig:** Erst nach P6-Skalen-Studie tackeln. Wenn P6 bei 100 Splats
schon bricht, ist N3DV-Diagnose verfrüht. Falsifikations-Test (oben)
priorisiert die Hypothesen.

---

## Risk-Register

| Risk                                      | Severity | Trigger                            |
|-------------------------------------------|----------|-----------------------------------|
| Manifold-Drift mit Euclidean+Renorm konvergiert nicht in <30k Iter | **Showstopper** | Bestätigt im Diagnose-Sprint ⇒ P5 vor P10 reaktivieren, projizierter Gradient oder volle Riemannsche Adam |
| Strichholz-Init strukturell unfixbar bei großen t | hoch     | Falsifikations-Test zeigt rank-1, (p,q)-Perturbation reicht nicht ⇒ Reformulierung der φ_t-Skalierung nötig |
| σ_k API-Split bricht externe Caller       | medium   | API-Bruchstelle, alle Test-Migrationen + train_n3dv.py + initialization.py |
| GPU-Pfad weicht von Toy ab                | medium   | Numerischer Vergleich zeigt L1 > 1e-2 ⇒ Inria-Adapter-Bug |
| Density-Schedule + Manifold instabil      | medium   | Splat-Anzahl explodiert, oder min_c < -0.9 nach Density-Events |

---

## Was als nächstes sinnvoll wäre

Reihenfolge folgt der Logik "schließe Validierungs-Achse von unten nach
oben", mit Parallelisierung wo möglich:

1. **Falsifikations-Test σ_αα = 1e-8** (~1h)
   60-Sek-Render plus Setup. Eliminiert Hälfte der P10-Hypothesen vor
   allen anderen Schritten. **Höchste Priorität.**

2. **Antidiagonal-Logging einbauen** (~1h)
   In `training.py` pro Iter loggen:
   `min_c`, `max_c`, `mean_c`, `cond_sigma_k`, `max_grad_sigma_bb`,
   `n_active`, `sigma_bb_pct`. Schwellen-Warnings bei
   `min_c < -0.9`, `max_c > 0.99`, `cond_sigma_k > 100`,
   `max_grad_sigma_bb > 1e6`. Ab jetzt jede Diagnose mit diesen Zahlen.

3. **End-to-End gradcheck für P4** (~30 min)
   `torch.autograd.gradcheck` mit double precision auf
   `compute_derived ∘ condition_on_time → Render → MSE-Loss`.

4. **σ_k API-Split** (~halber Tag)
   Touchpoints: `gaussian.py` (Field + temporal), `rasterizer.py` (pixel),
   `initialization.py` (defaults+passthrough), `trainable.py` (parameter
   machinery + lr group), `train_n3dv.py` (call sites), plus
   `test_rendering.py` und `test_initialization.py`. API-Bruchstelle —
   alle externen Caller migrieren.

5. **P3 GPU-Verifikation** (~4h, parallel zu 1–4 möglich)
   `is_available()` aktivieren, numerischer Vergleich CUDA vs. Toy mit
   Toleranz auf 100 zufälligen Inputs, plus visueller A/B.

6. **P6 Skalen-Studie** (~1 Tag, parallel zu 5 startbar bei ≤1k Splats)
   100 → 1k → 10k Splats auf Synthetik, Konvergenz und Antidiagonal-
   Verhalten. Ergebnis sagt, wo das System bricht.

7. **P10 Diagnose-Sprint (a''')(b)(c)** (~1 Tag)
   Erst sinnvoll, wenn P6-Skala klar ist. (a''') = (p,q)-Perturbation
   um 25–30°, nicht σ_αβ = 0.001.

8. **Open Items für später:** Adam-Moment-Erhaltung bei Density-Ops,
   Opacity-Reset, Antidiagonal-Stress-Test in P5, Riemannsche-Strategie-
   Vergleich (b)/(c) wenn Risk-Item triggert.

---

## Empfohlenes Diagnose-Logging im Trainings-Loop

```
min_c, max_c, mean_c   = p·q-Statistiken über alle Splats
cond_sigma_k           = max-eigval / min-eigval von Σ_k (lokal),
                         erfasst Strichholz-Topologie
max_grad_sigma         = max over splats of |∂L/∂σ_ββ|
mean_alpha_eff         = mean over splats of α_eff(t₀)
n_active               = #splats with α_eff > ε
sigma_bb_pct           = percentile(σ_ββ, [10, 50, 90])
```

Schwellen:
- `min_c < -0.9` ⇒ Antidiagonal-Approach
- `max_c > 0.99` ⇒ Diagonal-Approach (späte-Frame-Strichholz)
- `mean_c` Drift Richtung 1 oder -1 ⇒ Population als Ganzes wandert
- `cond_sigma_k > 100` ⇒ Strichholz-Topologie
- `max_grad_sigma > 1e6` ⇒ numerische Instabilität
- `n_active < 10%` außerhalb Opacity-Reset ⇒ Splat-Collapse

## Visualisierungen pro Phase

`visualize_phase{0..7}.py` als Regression-Schutz pro Phase. Für P6/P10
zusätzlich systematische **Time-Lapse pro Iter-Bucket** (iter 1k, 5k, 10k,
30k) committen. Der gegenwärtige Diagnose-Stand existiert nur als 2 PNGs
(iter 0, iter 100); eine systematische Time-Lapse hätte die Strichholz-
Hypothese früher sichtbar gemacht.