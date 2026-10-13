# TODO

Goal: false-positive-free, high-recovery flare detection across a wide range
of light-curve morphologies. Every change is judged by injection-recovery
(`notebooks/injection_recovery.py`, 100 light curves, seed 92, 787 flares)
and its flare-finding-only twin with perfect detrending
(`notebooks/flare_finding_only_injection_recovery.py`).

## Current reference (2026-10-13)

| | Recovered | S/N 2–5 | S/N 5–10 | S/N > 10 | False positives |
|---|---|---|---|---|---|
| Pipeline (`diag_plots/`) | 748 (95.0 %) | 138 / 176 | 197 / 198 | 413 / 413 | 3 (1 outside injected flares) |
| Perfect detrending (`diag_plots_flare_finding_only/`) | 760 (96.6 %) | 149 / 176 | 198 / 198 | 413 / 413 | 1 (0 outside injected flares) |

Settings behind these numbers:

- Detrending: `custom_detrending` defaults (polynomial baseline, GP with
  second-period and residual-period terms, 2-day minimum damping time,
  matched filter iterated 3 times).
- Detection: `find_flares` defaults (since 2026-10-13): `max_window=10,
  merge_sigma=0.25, extend_sigma=1.0, beta_window_days=1.0`, i.e. the window
  detector with a 1-day local beta. `injection_recovery.detect_flares` passes
  the same values explicitly.
- Matching (`fakeflares.match_flares`): an injected flare counts as recovered
  if its peak lies inside a detection; a detection containing no injected
  peak is a false positive, even if it lies on a flare's tail.

Before the window detector (classic 3 x 3 sigma rule): 662 (pipeline) and
692 (perfect detrending), 0 false positives outside injected flares.

## Open

- **LC 21, t = 18.64 d** is the one false positive outside injected flares:
  a 20-min, 1–2 sigma bump at a turning point of the fastest, strongest
  modulation in the sample (P_min 17 h). The GP (2-day minimum damping
  time) is slightly too smooth there, and the window detector sums the
  bump. The local beta removed the other turning-point and edge cases but
  not this one.
- **Tutorial output.** `find_flares` now defaults to the window detector;
  the flare tables printed in `01_Getting_Started.rst` and the other
  tutorials were made with the classic rule and need regenerating.
- **`merge_sigma=0.25`** joins fragments when the flux between them averages
  at least 0.25 sigma, which reliably reattaches the faint tails of bright
  flares (otherwise counted as separate events). It can also join two
  separate flares close in time into one detection; not yet quantified.
- **Remaining misses** are at the detection limit: 38 of 176 flares at
  S/N 2–5 with the pipeline (27 with perfect detrending). The detrending
  now costs 12 flares relative to perfect detrending (7 of them FWHM
  <= 15 min, 7 FWHM 120–200 min).

## Failing tests

The full suite: 244 passed, 4 failed.

- **Quantity vs plain number** (4 tests in
  `altaipony/tests/detrending/test_pipeline.py`): `custom_detrending`
  returns `detrended_flux` as an astropy Quantity labelled e-/s (but holding
  relative flux), and these tests combine it with plain numbers
  (`UnitConversionError`). Decide what type `detrended_flux` should have
  (review plan, step 1), then fix the tests or the code.
  - `TestCustomDetrending::test_measure_flare_basic`,
    `test_measure_flare_has_required_columns`,
    `test_measure_flare_calculates_duration`: add `500` to
    `flc.detrended_flux`.
  - `TestCustomDetrending::test_custom_detrending_removes_variability`:
    compares `np.nanstd(detrended_flux)` with the std of the plain flux.
- **Flaky:** `test_flarelc.py::test_sample_flare_recovery` fails about one
  run in six. It injects unseeded random flares; sometimes one lands next to
  the mock's real flare (cadences 15–18) and merges with it.

## Other bugs

- `FlareLightCurve.find_cont_windows` raises `ValueError: flux and flux_err
  must have the same units` on the light curve built in
  `test_flarelc.py::test_detrend`. `find_iterative_median(per_window=True)`
  would still hit it there. It may come from `flux_err` being multiplied by
  e-/s twice somewhere.
- `fit_flares.py` passes its fitted "fwhm" to `flare_model_davenport2014`,
  whose third argument is a duration (FWHM = duration / 2), so the FWHMs it
  reports are twice the actual FWHM.
- The matched filter's Davenport template fits injected Mendoza flares best
  at about twice their FWHM (same convention mismatch). Only the mask extent
  depends on it.
- `fakeflares.flare_model_mendoza2022(..., upsample=True)` raises
  `ValueError: The smallest edge difference is numerically 0.` whenever
  `tpeak` coincides with a cadence. `altaipony.synthetic` does not use
  `upsample`.

## Not yet tested (robustness rather than tuning)

The sample has 25-day, 2-min light curves with spot modulation, white noise
and flares only.

- **Light-curve length.** The second-rotation term is weaker on 8–12-day
  light curves (review plan, step 9); test single TESS orbits (~13 d) and
  longer Kepler/K2 stretches.
- **Other cadences.** 20-s and 30-min data: polynomial windows, GP binning
  and matched-filter templates are tuned on 2 min. The window detector's
  beta window is set in days (`beta_window_days`), its window lengths (3–10
  cadences) are not.
- **Real-data systematics.** Scattered light, momentum dumps and other
  short-timescale trends are not in the synthetic data; the window detector
  will be more sensitive to them than the classic rule. Test on real TESS
  light curves of quiet stars.
- **Complex flares** (two flares overlapping in time) and flare rates much
  higher than ~0.3 per day.

## Open, related

- `fit_flares.py` fits with the Davenport et al. (2014) template; switch it
  to Tovar Mendoza et al. (2022), which injection-recovery uses (and fix the
  FWHM convention above). `matchedfilter.py` also uses Davenport.
- `detrended_flux` is relative flux but labelled e-/s (review plan step 1).
- `injrecanalysis.py` works on `FlareLightCurve.fake_flares`; the benchmark's
  results table uses the same column names (`peak_time`, `duration_d`,
  `amplitude`, `ed_inj`, plus the finder's columns), so its tiling and
  heatmap functions can be used on it.

## Resolved

Detection and injection-recovery (2026-10-12/13):

- `find_flares` defaults to the window detector with merging and extension
  (2026-10-13); the classic rule is `max_window=None, merge_sigma=None,
  extend_sigma=None`. Changelog entry added.
- **Window detector** (`altai.find_flares(max_window=10)`): flags windows of
  3–10 cadences whose combined probability of being noise (Fisher's method,
  each point capped at 3 sigma) is below that of 3 points at 3 sigma. At
  n = 3 it reduces to the classic rule; 10 points need about 1.8 sigma each.
  Pure white noise: 1 of 200 light curves of 18,000 cadences gets a
  detection. With perfect detrending: 770 recovered (from 692), no detection
  on noise. Two additions made it work in the pipeline:
  - `merge_sigma` lowered from 1.0 to 0.25: the detector found the faint
    tails of bright flares separately (146 tail fragments counted as false
    positives with perfect detrending at 1.0, 11 at 0.25).
  - **Local beta** (`_local_beta`): for windows longer than 3 points, the
    excursions are divided by how much more n-cadence block means scatter
    within about 1 day than white noise would (robust MAD, at least 1).
    This suppresses GP residuals that are correlated over tens of minutes
    (segment edges, turning points of fast modulation): false positives
    outside flares 7 -> 1, recovery 760 -> 749 (now 748 after the fix
    below). Window length 1 day beat 0.5 day (731 recovered, 1 false
    positive) and 2 days (750, 3).
  - A unit test caught that windows of exactly 3 points never fired (three
    capped points sit exactly on the threshold, compared with `>`); fixed
    with `>=` and a small tolerance.
- **Detection extension** (`find_flares(extend_sigma=1.0)`): each detection
  grows over adjacent cadences above 1 sigma, so detections that started on
  the decay of a faint flare reach back over its peak. With the classic
  rule: 646 -> 657 recovered, false positives 13 -> 2, none lost.
- **Noise estimate at segment starts** (`noise.estimate_detrended_noise`):
  when a segment starts during a flare, the rolling std was NaN there and
  only filled forwards, so no flare could be found in those cadences. Now
  filled in both directions (+2 bright flares, LC 7 and 57).
- **Matching on the peak** (`fakeflares.match_flares`, also used by
  `FlareLightCurve.sample_flare_recovery` instead of
  `merge_fake_and_recovered_events`): recovered = injected peak inside a
  detection. The benchmark used to accept any overlap with the flare's
  10 % window.
- The benchmark's results table has the column names of
  `FlareLightCurve.fake_flares` and carries the finder's full row per
  matched detection; false positives are saved to CSV too.
- `fakeflares.template_equivalent_duration` computes the injected ED for
  both `inject_fake_flares` and `synthetic.inject_flares` (the template's ED
  is about 2.0 `ampl * fwhm`, not 1.82 as documented before).
- `fakeflares.mod_random` draws from its own `numpy.random.Generator` and no
  longer reseeds numpy's global random state.

Detrending:

- **Iterated matched filter** (`matched_filter_iters=3`): each pass runs on
  the residual of the latest GP refit, where the flares already masked are no
  longer absorbed: 657 -> 662 recovered, no extra false positives. With a
  perfect flare mask (the injected windows) the pipeline recovers 695, the
  whole gap to perfect detrending; all remaining detrending losses were
  flares the matched-filter mask did not cover.
- The GP runs for every star (spotless ones too: 133 vs. 119 of 164
  flares); the periodogram, computed without the baseline's flare mask,
  only provides its starting period, and the FAP gates the second-period
  search and the residual-period term.
- Residual-period term (`gp_residual_term=True`): an extra rotation term at
  a significant period left in the GP residual (often an uncovered
  harmonic, e.g. 0.346 d on LC 21).
- Second rotation term (`gp_second_rotation=True`) for a second, unrelated
  spot period.
- Matched filter runs on the residual of a first GP fit, not on the baseline
  residual (which masked rotation maxima on fast rotators).
- `custom_detrending` returns relative flux with a quiet baseline of exactly
  1, independent of input units; the GP is fitted on median-normalised flux.
- `find_iterative_median` computes one global median by default.
- Removed without loss: the spline, multi-sine and `auto` baselines
  (60–115 false positives vs. 7 for the polynomial), the Savitzky-Golay
  passes and the whole Savitzky-Golay mode (`savgoldetrending.py`,
  `detrend("savgol")`), the 6–36 h sinusoid correction, the S/N periodicity
  gate.
- Fixed earlier: GP predictive variance no longer computed by default
  (out-of-memory kills); `fakeflares.flare_eqn` NaNs long before the peak;
  `flux_err` not restored before the GP.

## Tried and dropped

All with injection-recovery (100 LCs, seed 92); numbers relative to the
reference at the time.

Detection:

- Global beta (one per segment): removed the edge false positives but not
  the turning-point bumps, and cost 22 short flares with perfect detrending
  (it also made the 3-point windows stricter than the classic rule).
- `extend_sigma=2`: 681 vs. 692 (perfect detrending), 14 vs. 3 false
  positives.
- `merge_sigma` 0.5 with the window detector: 21 vs. 11 false positives
  (tail fragments) at equal recovery.

Matched filter and GP flare mask:

- Matched-filter threshold S/N 4 instead of 5: identical results. Most lost
  flares were already detected; their masks were too small.
- Growing the matched-filter mask down to 1.0 or 0.5 sigma (`grow_sigma`):
  +6 / +9 flares, but 2 false positives on noise (mask runaway on fast
  rotators).
- Masking the best-matching template's extent (decay to 20 % or 10 % of the
  peak): 690 recovered, but 18–25 false positives on noise. The filter also
  flags rotation leftovers, the wide masks hide modulation maxima from the
  GP, and the passes snowball at mask edges.
- Guards for that: a shape test (flare vs. symmetric Gaussian template) and
  hiding masked cadences from later passes. They removed most of the runaway
  but rejected real flares too (the first GP's residual makes partly
  absorbed flares look symmetric): 666 with the template extent, 655
  without it, false positives on noise in both.
- Earlier: other FWHM grids for the matched filter; tail extension and 2σ
  hysteresis extension of the baseline mask (false positives on fast
  rotators); a 2.5σ baseline mask (+3 flares, 1 false positive on noise);
  subtracting fitted flares before the GP; doubled polynomial windows.

GP:

- No post-fit clipping (`gp_clip_iters=0`): removes almost all window-
  detector false positives (the clip pulls a slightly low model further down,
  most at segment edges), but loses 40–49 weak long flares, which the clip
  masks at the binned resolution. Not an option.
- Clipping relative to each point's own error (binned vs. unbinned edge
  points): no measurable effect on the edge sag.
- Protecting training points within 0.1 d of segment edges from the clip:
  fixed the three edge false positives but cost 3 flares and moved small
  residuals elsewhere (2 new false positives mid-segment).
- Binning by 2 instead of 10 (−31 flares); minimum damping time 1 or 5 days
  instead of 2 (no better); a non-periodic SHO term (could not follow a
  second fast mode).
