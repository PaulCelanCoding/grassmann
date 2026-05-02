# Grassmann Splatting — Phasenplan

Inkrementelle Implementierung mit testbaren Invarianten. Pro Phase werden
zwei Achsen unterschieden:

- **Implementiert:** Code existiert, prüft Mathematik-/API-Konsistenz
- **Validiert:** empirisch verifiziert auf realistischer Skala (Konvergenz,
  Stabilität, korrekte Outputs auf echten Daten)

**Eine Phase ist erst abgeschlossen, wenn beide Achsen grün sind.** Das
unterscheidet sich vom bisherigen Test-Apparat, der primär Achse 1 prüft.

**Aktueller Stand:** P0–P5 vollständig (beide Achsen). Ab P6 driftet die
Validierung — Tests laufen grün, aber das System konvergiert auf echten
Daten nicht. Die Aufgabe der nächsten Iteration ist, die Validierungs-Achse
schrittweise zu schließen.

---

## Was die existierenden 113 Tests *nicht* leisten

Bevor die Phasen kommen, der wichtige Disclaimer: Tests wie
`test_init_gaussian_mean_matches_point_approximately` prüfen, ob ein
Gaussian an einer 3D-Position platziert wird (ja, korrekt). Sie prüfen
**nicht**, ob diese Platzierung als Trainings-Initialisierung sinnvoll ist
(nein — es ist ein rank-1 Strichholz entlang des Sichtstrahls). Analog für
Rendering, Densification, Optimierung. Konsistenz der Mathematik ≠
Tauglichkeit des Algorithmus.

Validierungs-Tests müssen anders aussehen: nicht "tut der Code was er sagt",
sondern "produziert er auf realistischer Skala das gewünschte Ergebnis".

---

## P0 — Geometrische Primitive

**Implementiert (✅):** Quaternion-Algebra, `line_to_pq`, `pq_to_basis`.
`test_quaternion` (11) + `test_grassmann` (16) inkl. Issue-1- und
Issue-3-Tests.

**Validiert (✅):** Mathematik ist algebraisch — kein Skalen-Sprung möglich.
Algebraische Korrektheit = Funktionale Korrektheit.

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

**Validiert (✅):** rein algebraisch, plus Cross-Check gegen explizite 4D-
Marginalisierung.

---

## P3 — Forward-Render

**Implementiert (✅):** Toy-Rasterizer + Inria-CUDA-Adapter. `test_rendering`
(18), `test_rasterize_temporal_fade`, `test_rasterize_occlusion`.

**Validiert (⚠ teilweise):**
- Toy-Pfad auf Synthetik mit ≤ 100 Splats: ✅
- CUDA-Pfad auf realer GPU: **nicht ausgeführt** (`is_available()` returnt
  False im Test-Setup, alle 10 Tests laufen den Fallback)
- Bit-Vergleich Toy ↔ CUDA auf 100 zufälligen Inputs: **fehlt**

**Was fehlt:** GPU-Run mit 1k–10k Splats und Vergleich gegen Toy-
Referenz, plus Bit-Vergleich-Test als CI-Hook.

---

## P4 — Autograd-Backward

**Implementiert (✅):** alles pure PyTorch, Autograd. `test_initialization`
(21), `test_init_gaussian_mean_matches_point_approximately`.

**Validiert (✅):** gradcheck durchgelaufen auf compute_derived,
condition_on_time, und Komposition. **Aber:** gradcheck prüft, ob
analytische Ableitungen mit FD übereinstimmen — er prüft nicht, ob die
Ableitung *des richtigen Funktionals* berechnet wird. Die Mathematik selbst
ist korrekt, das gilt es nicht zu hinterfragen; eine eigene Validierung auf
echten Daten ist hier nicht nötig.

---

## P5 — Riemannsche Optimierung (Spielzeug)

**Implementiert (✅):** Euclidean Adam + Double-Renormalisierung
(`trainable.py:99-102`, `Trainer.renormalize_manifolds()`). Strategie-
Wahl bewusst pragmatisch.

**Validiert (✅ Toy, ⚠ Annahme nicht stress-getestet):**
- Single-Gaussian-Konvergenz auf Synthetik: ✅
- Multi-View-Multi-Frame mit 3 Splats: Loss 0.047 → 0.020 in 100 Iter,
  monotones Plateau danach
- **Antidiagonal-Stress-Test fehlt** (Init bei p · q = -0.95)
- **Strategie-Vergleich Euclidean vs. projiziert vs. Riemannsch fehlt**
  — wir wissen nicht, ob (a) auf realer Skala ausreicht oder ob (b)/(c)
  nötig werden. Das ist eine offene Forschungsfrage, kein Engineering-Item.

---

## P6 — Single-Scene Overfit, feste Splat-Anzahl

Hier beginnt die Validierungs-Lücke.

**Implementiert (✅):** `test_trainer_multi_view_multi_frame` läuft.

**Validiert (❌):** der existierende Test ist 3 Splats auf Synthetik. Das
ist kein Skalen-Test. Was fehlt:

- 100 Splats, 5–20 Views, 30k Iter ⇒ PSNR-Ziel
- Vergleich gegen Standard-3DGS auf identischer Synthetik (rank-3 ohne
  Manifold) ⇒ Qualitätsdelta
- Loss-Kurven-Inspektion mit Antidiagonal-Logging (siehe unten)

**Was wir aus dem N3DV-Versuch wissen:** der Sprung von 3 auf 30k Splats
bricht das System. Aber wir wissen nicht, ob es bei 100, 1000 oder erst
bei 30k bricht. **Diese Skalen-Studie fehlt komplett.** Sie wäre der
direkte Diagnose-Pfad: bricht es bei 100, ist die Init oder der σ_k-Bug
schuld; bricht es erst bei 10k+, ist es Densification oder
Riemannsche-Konvergenz.

---

## P7 — Densification (Mechanik)

**Implementiert (✅):** Clone, Split, Pruning manifold-erhaltend.
`test_density_control` (12).

**Validiert (❌):**
- "Synthetic Recovery" (zu wenige Splats ⇒ Densification rekonstruiert
  Detail) **ist nicht getestet als End-to-End-Konvergenz**, nur als
  Mechanik
- Splitting-Konsistenz im Render-Sinn (Summe der zwei neuen ≈ Original)
  fehlt
- Wir wissen nicht, ob das Manifold-Splitting in der Praxis Detail
  rekonstruiert oder nur Splats vermehrt

---

## P8 — Adaptive Density Control im Trainings-Loop

**Implementiert (⚠ partiell):**
- Density-Schedule existiert
- **Adam-Reset bei Density-Operationen** (Moments gehen verloren) —
  Standard-3DGS macht das partiell smarter
- **Kein Opacity-Reset** implementiert
- **Kein Antidiagonal-Logging** im Loop

**Validiert (❌):** noch nicht angefangen — siehe P10.

---

## P9 — Custom-CUDA-Backward

**Skip.** Wir nutzen Autograd durch `compute_derived`/`condition_on_time`,
nur der Rasterizer-Kernel ist CUDA. Custom-CUDA wäre spätere Performance-
Optimierung.

---

## P10 — Echte Daten (N3DV)

**Implementiert (✅):** Pipeline läuft.

**Validiert (❌, blockiert):** sichtbarer Collapse — radiale Streifen vom
Bildzentrum (`diagnose_render_iter100.png`).

Die Diagnose ist offen. Mehrere orthogonale Hypothesen, in absteigender
Wahrscheinlichkeit:

1. **Strichholz-Init (strukturell, kritisch).**
   `init_gaussian_from_point` mit σ_αβ = 0 produziert pro Punkt einen
   rank-1-Splat entlang des Sichtstrahls. J_embed hat Spalten [û | 0],
   also Σ_3D = û · σ_αα · ûᵀ. In jeder Kamera projiziert: Streifen.
2. **σ_k API-Doppelrolle (kritisch, Engineering-Bug).**
   Derselbe Parameter ist in `gaussian.py:155` temporale Varianz und in
   `rasterizer.py:80` 2D-Pixel-Varianz. Bei `sigma_k = 20` als Pixel-
   Varianz: Std 4.5 Pixel, Standard-3DGS-EWA ist 0.3 Pixel². Entwertet
   gleichzeitig das temporale Targeting.
3. **30k zufällige Punkte in 3m-Cube** statt COLMAP-Init.
4. **lr_pq = 1e-3** zu klein für Manifold-Reparatur.

**Diagnose-Sprint (orthogonal, einzeln testen):**
- (a) `init_gaussian_from_point(σ_αβ = 0.001)` ⇒ c_world ≠ 0,
  Conditioning shiftet+shrinkt
- (b) `sigma_k_pixel = 0.3` separat von `sigma_k_temporal` (API-Split)
- (c) COLMAP-`points3D.txt` als Init-Quelle

**Wichtig:** Erst nach P6-Skalen-Studie tackeln. Wenn P6 bei 100 Splats
schon bricht, ist N3DV-Diagnose verfrüht.

---

## Was als nächstes sinnvoll wäre

Die Reihenfolge folgt der Logik "schließe die Validierungs-Achse von unten
nach oben":

1. **P3 GPU-Verifikation** (~4h)
   `is_available()` aktivieren, CUDA-Pfad gegen Toy bit-vergleichen.
   Voraussetzung für jeden ernsten Skalen-Test.

2. **Antidiagonal-Logging einbauen** (~1h)
   In `training.py` pro Iter loggen:
   `min_c`, `max_grad_sigma_bb`, `n_active`, `sigma_bb_pct`. Schwellen-
   Warnings bei `min_c < -0.9`, `max_grad_sigma_bb > 1e6`. Ab jetzt jede
   Diagnose mit diesen Zahlen.

3. **σ_k API-Split** (~2h)
   Trennen in `sigma_k_pixel²` und `sigma_k_temporal²` mit eigenen
   Defaults. Migrations-Patch durch `gaussian.py`, `rasterizer.py`,
   `train_n3dv.py`.

4. **P6 Skalen-Studie** (~1 Tag)
   100 → 1k → 10k Splats auf Synthetik, jeweils Konvergenz und Antidiagonal-
   Verhalten. Ergebnis sagt, wo das System bricht und was P10 wirklich
   blockiert.

5. **P10 Diagnose-Sprint** (a)/(b)/(c) (~1 Tag)
   Erst sinnvoll, wenn P6-Skala klar ist.

6. **Open Items für später:** Adam-Moment-Erhaltung, Opacity-Reset,
   Antidiagonal-Stress-Test in P5, Riemannsche-Strategie-Vergleich
   (b)/(c).

---

## Empfohlenes Diagnose-Logging im Trainings-Loop

```
min_c           = min over splats of p·q
max_grad_sigma  = max over splats of |∂L/∂σ_ββ|
mean_alpha_eff  = mean over splats of α_eff(t₀)
n_active        = #splats with α_eff > ε
sigma_bb_pct    = percentile(σ_ββ, [10, 50, 90])
```

Schwellen: `min_c < -0.9` ⇒ Antidiagonal-Approach;
`max_grad_sigma > 1e6` ⇒ numerische Instabilität;
`n_active < 10%` außerhalb Opacity-Reset ⇒ Splat-Collapse.

## Visualisierungen pro Phase

Pro Phase ein PNG-Snapshot committen (`visualize_phase{0..7}.py` ist
bereits das richtige Schema). Spätere Patches, die einen Phase-Test grün
lassen aber visuell etwas brechen, sind so sofort zu sehen.