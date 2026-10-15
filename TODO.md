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

**1000 light curves** (seeds 92–1091, 7542 flares, same settings,
2026-10-13; scratch runs, not in `notebooks/`):

| | Recovered | S/N < 3 | 3–5 | 5–10 | > 10 | False positives outside injected flares |
|---|---|---|---|---|---|---|
| Pipeline | 7158 (94.9 %) | 54 % | 86 % | 99.1 % | 100 % | 62 (60 without flare flux, in 31 LCs) |
| Perfect detrending | 7254 (96.2 %) | 64 % | 90 % | 99.9 % | 100 % | 10 (9 tails of bright flares, 1 on noise) |

- 46 of the pipeline's 60 false positives without flare flux are on fast
  rotators (P_min < 24 h), mostly with low noise (< 500 ppm); four light
  curves (779, 811, 272, 880) have 27. Spotless stars have none. They last a
  median of 60 min: detrending residuals that the local beta does not
  suppress when they repeat every few hours.
- The detector itself gives 1 false positive on noise in 1000 light curves.
- Tails of bright flares (S/N 20–50), 5–13 FWHM after the peak, are
  sometimes detected separately from the flare (9 with perfect detrending, 2
  with the pipeline); for real data this would count one flare twice.

Before the window detector (classic 3 x 3 sigma rule): 662 (pipeline) and
692 (perfect detrending), 0 false positives outside injected flares.

**5σ initial mask** (`second_pass_sigma=5.0`, now `final_mask_sigma`, default since 2026-10-13; 1000
light curves): 7166 recovered, 45 false positives, 29 outside injected
flares. Who produces them (2026-10-14):

- 21 at segment edges (< 0.1 d), in 14 light curves: high amplitude over
  noise (median 810) and low noise (median 300 ppm). Light curves with
  amplitude/noise > 300 are 7 % of the sample but give 18 of the 21; 20 %
  of them have one.
- 16 on tails of injected flares (detector, not detrending): noisy (median
  1180 ppm), low amplitude/noise (median 19).
- 8 mid-segment, in 7 light curves: mostly fast rotators (median P_min
  9 h, amplitude/noise 100–650; LCs 811, 567, 261, 742) where the GP is
  slightly too smooth at sharp turns, plus two 14-min bumps on noise (LCs
  55, 897).
- Spotless stars and amplitude/noise < 100 have no edge false positives.

**GP noise model** (`gp_jitter_only=True, gp_jitter_floor=3`, default since
2026-10-14; 1000 light curves, scratch run): **7172
recovered, 33 false positives, 19 outside injected flares** (in 15 light
curves), 14 on flare tails. Recovery by S/N: < 3: 58.8 % (was 55.6 %),
3–5: 85.9 % (86.5 %), > 5 unchanged.

- Cause: the old jitter bound, 0.05 x the flux's standard deviation
  (modulation included), plus the benchmark's flat 1000 ppm `flux_err`.
  On quiet, strongly spotted stars the GP assumed ~10x the true noise
  (LC 272: 2749 ppm fitted, 298 ppm true), so it was too stiff at segment
  edges and turning points.
- The jitter is now fitted relative to the white noise from second
  differences, with a floor of 3x that noise. Without a floor (0.1) the GP
  absorbed flares on fast rotators (screen of 106 LCs: −25 flares, 19 of
  them at P_min < 12 h); floor 2: 12 false positives / 649 recovered,
  floor 3: 14 / 660, old model: 29 / 654.
- Churn on 1000 LCs: 24 false positives gone (16 LCs), 14 new (11 LCs),
  recovery +88 / −82. New: LC 966 (4; P_min 6.6 h, 279 ppm, largest
  binning bias), LC 779 (1 → 2), nine single ones.
- With the jitter at the noise, a single rotation term follows the second
  mode in `test_gp.py`'s two-period star. Check whether the second-rotation
  and residual-period terms still earn their keep.

**Bin factor** (`gp_bin_max_bias`, off by default): binning by 10 averages
a strongly curved modulation over 20 min, which shifts the binned points by
more than their (sqrt(10)-smaller) error on 27 of 1000 light curves. The
rule lowers the bin factor to 5, 3, 2 or 1 where the first GP fit, binned,
shifts by more than 0.5 binned sigma. On the old noise model (56 LCs): 22
vs. 29 false positives, −6 flares, unchanged where k stays 10. On top of
the new noise model (67 LCs: those 56 plus the 11 with new false
positives; the rule lowered k in 17): 13 vs. 19 false positives (LC 966
4 → 0, LCs 172 and 918 1 → 0), 4 vs. 5 on tails, 360 vs. 367 flares
(LCs 344 −2; 125, 387, 649, 714, 966 −1, all fast rotators). Default
since 2026-10-14 (`gp_bin_max_bias=0.5`).

**Current default, 1000 light curves** (jitter floor 3 + bin rule, scratch
run, 2026-10-14): **7165 recovered, 26 false positives: 13 outside injected
flares (in 12 LCs), 13 on flare tails.** Recovery by S/N: < 3: 58.5 %,
3–5: 85.5 %, 5–10: 99.1 %, > 10: 100 %. Against the jitter-only run
exactly as the screen predicted: −6 false positives (LCs 966, 172, 918),
−7 flares (6 fast rotators), nothing else changed. Against the 5σ-mask
reference: 13 vs. 29 outside flares, 13 vs. 16 on tails, 7165 vs. 7166.
**Since then** (2026-10-15, 1000 light curves each): one polynomial pass
instead of two: 7165 recovered, 14 outside flares, 12 on tails (no change
beyond churn; removed together with the no-GP path). Preliminary mask
without `flux_err` in its threshold (true 5 sigma): **7170 recovered, 13
outside flares (12 LCs), 12 on tails** -- current reference.
Open: the remaining ones (LC 874 edge, short edge ones, 10–20 min noise
bumps in noisy stars above the 1-in-1000 rate of perfect detrending).

## Open

- **Missing cadences.** `custom_detrending` no longer interpolates missing
  cadences (2026-10-13). Test with a benchmark variant that drops 1–2 % of
  the cadences at random plus a few short dropouts; if recovery worsens,
  bring back an interpolation (with the pipeline's own windows, filled from
  the time stamps, quality 0 on filled cadences).
  `FlareLightCurve.interpolate_missing_cadences` was removed: it replaced the
  caller's `cont_windows`, dropped all but four columns, interpolated quality
  flags as numbers and did nothing without a `cadenceno` column.
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

The full suite: 263 passed (2026-10-15). The 4 Quantity failures are fixed:
`detrended_flux` is dimensionless relative flux.

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

`FlareLightCurve` and `lcio` (2026-10-13):

- `find_flares` returned the previous result unchanged when flares had already
  been found, ignoring new settings; it now always searches anew. Its
  continuous-window gap and median iterations are options (`maxgap`,
  `median_iterations`).
- `to_flare_lightcurve` changed the class of the caller's light curve; it now
  converts a copy.
- `read_from_fits` read BTJD/BKJD times as JD (dates in 4708 BC) and refused
  files without TARGETID.
- `repr` (and `save_to_fits`' default name) crashed without a numeric cadence.
- `sample_flare_recovery(save_lc_to_file=True)` wrote to a literal
  `{folder}before.fits` and referenced `self.targetit`; `detrend_kwargs` had a
  mutable default.
- The fake-flare tutorial imported a non-existent `lcio.from_mast`; it now
  uses lightkurve's search and `to_flare_lightcurve`.

- Copy semantics: methods that analyse the light curve work in place and
  return it (`find_flares`, `detrend`, `find_cont_windows`,
  `mark_flagged_flares`, `get_saturation`, `characterize_flares`,
  `sample_flare_recovery`, `fit_flares`, `load_injrec_data`); only
  `inject_fake_flares` returns a new light curve. Before, some copied, some changed the light curve, and `fit_flares`
  returned None.

- `flc.detrend()` uses `custom_detrending` by default (keyword arguments
  passed on, `func=` for other functions); `mode` is deprecated. The
  tutorials use the short form (the Getting Started one also passed
  `spline_coarseness=8`, an option of the removed spline baseline).

- Removed `get_saturation` (it needed K2 pixel data the class no longer
  carries), the K2SC and `george` dependencies and the K2SC citations;
  `progressbar2` replaced by a `tqdm` bar showing injected and recovered fake
  flares. `celerite2` was missing from the dependencies and is now listed.

- Reading an uncomputed column (`detrended_flux`, `detrended_flux_err`,
  `gp_model`, `it_med`) returns None instead of creating a NaN column; the
  steps that need them give a clear error. `inject_fake_flares` before
  detrending used the iterative median of the (then all-NaN) detrended flux;
  it now uses the raw flux's.

Still open from the same review (API, not bugs): plots of the detrended flux
that accept an `ax`.

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
- Two-sided post-fit clipping (`gp_clip_two_sided`, 2026-10-13): fixed some
  edge false positives on the offenders, but worse on the 1000 run. Removed.
- Not masking matched-filter events within 0.1 d of segment edges
  (`matched_filter_edge_days`): 39 vs. 45 false positives on 1000 light
  curves but −13 flares and 4 new false positives elsewhere. Removed.
- Padding segment edges with an extrapolation before the GP fit
  (`gp_edge_pad`, 2026-10-14): straight lines and quadratics, 20–50
  cadences, fitted on 1–3 pad lengths. On high amplitude/noise stars any
  pad of 5 % of the period misses the trend by 5–15σ (the cubic term), so
  the GP follows it and the matched-filter mask runs away. Capped at 20
  cadences with a quadratic: 22 vs. 29 false positives on the 21 offenders,
  but 1 new false positive in 50 clean controls and −2 flares; at best a
  wash. Removed. If it comes back: never pad a segment shorter than the
  fit window (LC 871, a 0.05-d segment, curled off at both ends), pad after
  binning, and keep padded points out of the clip and the edge-anchor
  protection.
- Binning by 2 instead of 10 (−31 flares); minimum damping time 1 or 5 days
  instead of 2 (no better); a non-periodic SHO term (could not follow a
  second fast mode).
