# Review plan: `altaipony/detrending`

A step-by-step read-through of the detrending code, ordered along the data
flow and organised around one question per step: *is the science sound?*
Every step ends with tests that pin down the behaviour you have convinced
yourself of. The tests go into `altaipony/tests/detrending/`.

Line numbers refer to commit `2ab3376`.

**How to use this plan**

- Work through the steps in order. Each step has three parts:
  - **Read**: the code to go through.
  - **Check**: the questions to answer. Items marked ⚠ are suspicions I noticed
    while restructuring. They are hypotheses to confirm or refute, not bugs.
  - **Tests**: what to write once you understand the behaviour.
- When a check turns up a real problem, first write a test that shows it
  failing, then fix the code, then watch the test pass.
- Tick the boxes as you go.

---

## Step 0: Test setup

- [x] Create `altaipony/tests/detrending/` with `__init__.py` and a
      `conftest.py` that provides `make_lc(...)`, built on
      `altaipony.synthetic`:
  - `make_time_grid(duration, cadence_min, gaps=[(start, length), ...])`
  - `sinusoidal_variability(time, modes=[dict(period=..., amplitude=...)])`,
    each mode optionally with a P/2 harmonic (`harmonic_ratio`,
    `harmonic_phase`); fast rotators are modes with short periods
  - `inject_flares(time, flux, flares=[(tpeak, fwhm, ampl), ...])`, using the
    Tovar Mendoza et al. (2022) template. Its peak is ~0.95 `ampl`, its
    measured FWHM ~1.09 `fwhm`, and its equivalent duration
    ~1.82 `ampl * fwhm`.
  - add white noise and a `scale` factor (to mimic unnormalised e-/s flux) in
    `make_lc` itself. (`generate_synthetic_lc` already returns e-/s flux,
    with a quiescent baseline drawn uniformly from 1000–100000 e-/s,
    `baseline_range=None` for normalised flux; `inject_flares` takes
    `baseline=` and keeps amplitudes relative.)
  - Also provide a `FlareLightCurve` fixture built from it.

  Done: `make_lc(duration, cadence_min, gaps, modes, flares, noise, scale,
  seed)` returns `(lc, flare_table)` and is available as a
  session fixture `make_lc`; `synthetic_lc` is the default light curve
  (8 days, e-/s at 3400, spots, a gap, two flares). `test_make_lc.py`
  checks it.
- [x] Register a `slow` marker (done in `conftest.py`) for tests
      that run the full pipeline. Keep the other tests short: a few thousand
      cadences, a few seconds each.
- [ ] Run command, memory-capped:
  ```bash
  MPLBACKEND=Agg systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
      python -m pytest altaipony/tests/detrending -m "not slow"
  ```

## Step 1: The pipeline's contract

**Read:** `pipeline.py`, `custom_detrending` (34–386) top to bottom, without
going into the helpers yet; then `_apply_gp_step` (389–502).

**Check:**
- What exactly does `custom_detrending` promise to return? Write it down in
  one sentence. It should cover what `detrended_flux` means, its zero point,
  its units, and which other attributes (`it_med`, `gp_model`) are set.
- ⚠ **Zero point depends on the path.** With the GP (periodic stars),
  `detrended_flux = flux − gp + offset + 1.0` (492), so the baseline is at
  **1.0**. Without the GP, it stays at the original flux level (savgol path).
  Is this inconsistency intended?
- ⚠ **Units and scale.** In the GP path, `detrended_flux` is
  "1 + residual in the flux's own units", but it is labelled e-/s (493). For
  flux in e-/s (e.g. a baseline of 3400), a 1 % flare becomes a residual of
  34 on top of 1.0. Do relative amplitudes and equivalent durations computed
  downstream (`it_med`, `find_flares`, `measure_flare`) then depend on the
  absolute flux level? The injection-recovery runs never test this, because
  their flux is normalised to ~1.
- Which steps only produce *masks*, and which produce the *output*? In the GP
  path, the baseline and Savitzky-Golay results only feed the GP's flare mask.

**Done so far** (`tests/detrending/test_pipeline.py`): baseline at exactly
1 and scale invariance for both the GP and the non-GP path; relative
amplitudes from `find_flares`. Fixed along the way: normalisation of both
paths, GP fitted on normalised flux, `flux_err` restored before the GP.

**Tests** (`test_pipeline.py`):
- [ ] Output has the same length and time stamps as the input, keeps NaNs
      where the input had them, and sets `it_med` and `gp_model`.
- [x] Zero point: a flat, flare-free stretch of `detrended_flux` sits at the
      documented level, **for both** the GP path (periodic star) and the
      non-GP path (`use_gp=False` or a non-periodic star).
- [x] **Scale invariance:** the same light curve with `scale=1` and
      `scale=3400` gives the same *relative* flare amplitude and the same
      equivalent duration. I expect this to fail today if the ⚠ above is real.
- [ ] Fold in the 4 Quantity failures from `TODO.md`
      (`test_measure_flare_*`, `test_custom_detrending_removes_variability`)
      once you have decided what type `detrended_flux` should have.

## Step 2: Gaps and cadence

**Read:** `pipeline.py` 230–240.

**Check:**
- ⚠ `dt = np.mean(np.diff(time))` (232) includes the gaps. A single 1-day
  gap in 25 days of 2-min data raises the mean step noticeably. That inflates
  `maxgap * dt`, the gap threshold, and the Savitzky-Golay window lengths
  that are computed from `dt`. Should this be the median?
- What does `interpolate_missing_cadences` fill in, and is every
  interpolated cadence excluded from fits and from flare detection?

**Tests** (`test_pipeline.py`):
- [ ] A light curve with one long gap is split into exactly two segments, and
      the Savitzky-Golay window lengths equal those of the same light curve
      without the gap.
- [ ] No cadence that was interpolated into a gap ends up in a detected flare.

## Step 3: Detecting the rotation period

**Read:** `periodicity.py` (`lomb_scargle` 7–56,
`detect_strong_periodicity` 59–160); call site `pipeline.py` 246–261.

**Check:**
- Search range is 0.1 d to half the light curve. What happens with periods
  near the ends of that range?
- ⚠ The Baluev false-alarm probability assumes white noise. With red noise,
  flares or spot evolution it will be tiny almost always: all 100 synthetic
  light curves came out "periodic". Is the FAP doing any work, or does the
  amplitude threshold decide everything?
- The amplitude comes from the single best-fit sine, relative to the median
  flux. For non-sinusoidal spot curves, does that under- or overestimate?
- Can a large flare create a significant peak?

**Tests** (`test_periodicity.py`):
- [ ] A pure sine with P ∈ {0.3, 2, 8} d is recovered to within 1 %, and its
      relative amplitude to within 5 %.
- [ ] White noise only → not periodic.
- [ ] Flares on white noise, no modulation → not periodic.
- [ ] Light curve shorter than 2 × `period_min` → not periodic, no error.
- [ ] `lomb_scargle` drops non-finite points and never searches periods
      longer than the data span.

## Step 4: Choosing the baseline

**Read:** `pipeline.py` 256–305 (the branch on `baseline_method` and
`use_multisine`).

**Check:**
- The decision is: detrender if requested; else multi-sine if periodic and
  P < 5 d; else spline. Is 5 d justified? What happens to a periodic star
  with P = 4.9 d vs 5.1 d?
- `flare_finder_altai` uses the polynomial baseline ("detrender") for
  everything. Should that be the default instead of `"auto"`?

**Tests** (`test_pipeline.py`):
- [ ] `best_params["method"]` matches the expected branch for: non-periodic,
      periodic with P = 1 d, periodic with P = 8 d, and
      `baseline_method="detrender"`.

## Step 5: Polynomial baseline (`baselines/polynomial.py`)

The largest module. Review it in five parts.

**5a Preliminary flare mask:** `preliminary_flare_mask` (342–397),
`compute_rolling_local_sigma` (169–207).
- Is a 5σ cut on a rough 2-day polynomial residual conservative enough that
  real variability is never masked as flares?

**5b Window fits and blending:** `fit_window_grid` (400–516),
`run_all_windows` (519–578), `score_segment` (617–660),
`assemble_combined_trend` (663–757).
- Degree-4 polynomials in 0.4/0.6/0.8 d windows, on normal and half-shifted
  grids. Which variability timescales can they follow, and which flare
  durations would they eat into? A flare with FWHM ~1 h is a sizeable
  fraction of a 0.4 d window.
- Edge points are up-weighted 8× (`n_edge`, `edge_weight`). Why?
- `score_segment` mixes R²_adj, |log χ²_red|, masked fraction and point
  count with weights 0.15 / 0.30 / 0.05. The weights are heuristic. The
  softmax over scores then decides how sharply one fit wins. Is that
  sensible?

**5c Second pass:** in `run_detrending` (944–1165), the 3σ mask from the
first-pass residual and the refit.

**5d Rotation correction (6–36 h):** `multi_sinusoid_correction` (209–339).
- The greedy selection, the 2 % improvement threshold and the amplitude cap
  at 3σ. It matters for fast rotators: injection-recovery showed 48 false
  positives without it.

**5e Centring and final mask:** `center_residual_by_moving_average`
(786–833), `build_safe_final_flare_mask` (836–942).
- The final mask falls back to the previous one if more than 5 % of
  cadences get flagged. Is 5 % right for very active stars?

**Tests** (`test_polynomial.py`):
- [ ] Smooth trend (P = 3 d) + noise → residual scatter within 10 % of the
      injected noise.
- [ ] Flare preservation: inject flares with FWHM of 5 min, 30 min and 2 h on
      a smooth trend; the residual keeps ≥ 90 % of each peak amplitude (tune
      the thresholds once you know what the code can do).
- [ ] The rotation correction removes a 9 h sinusoid (residual scatter close
      to the noise) and does not fire on white noise.
- [ ] `build_safe_final_flare_mask` falls back to `previous_mask` when the
      candidate mask exceeds `max_mask_fraction`.
- [ ] `fit_lightcurve_detrender` maps results back onto the full grid with
      NaNs where the input had them.

## Step 6: Multi-sine baseline (`baselines/multisine.py`)

**Read:** `_segment_gaps` (9–49), `fit_multisine` (52–344).

**Check:**
- 5 harmonics, each with an amplitude that drifts as a polynomial in time
  (`amp_degree`), fitted per segment of ≤ `n_per` cycles. How many free
  parameters per segment, and can they start fitting flares?
- ⚠ One-sided clipping (only positive residuals) biases the fit *low* if
  noise above the model is also clipped. How large is that bias in units of
  σ?
- The per-segment period refinement uses ±5 % around P on a 400-point grid.
  Is that resolution enough?

**Tests** (`test_multisine.py`):
- [ ] A 3-harmonic spot curve (P = 1 d) is recovered; residual scatter close
      to the noise.
- [ ] One large flare changes the fitted model by less than 0.1σ away from
      the flare.
- [ ] Measure the bias from one-sided clipping on pure noise: mean residual
      versus 0.

## Step 7: Spline baseline (`baselines/spline.py`)

**Read:** `fit_spline` (11–113), `_fit_single_spline` (116–187),
`_build_knot_points` (225–297), `_sanitize_knots` (190–222),
`_evaluate_spline_fit` (300–406).

**Check:**
- ⚠ Knots sit at the **25th percentile** of each bin
  (`percentile_anchor=25`). On pure noise that puts the spline about
  0.67σ_bin below the true baseline. Does the later re-centring on `it_med`
  remove that offset completely?
- `_evaluate_spline_fit` scores candidates by MAD × (1 + asymmetry penalty +
  Gaussianity term + edge penalty). These are heuristics again; does the
  choice end up stable?

**Tests** (`test_spline.py`):
- [ ] Fix or rewrite the **12 failing spline/knot tests** from `TODO.md`.
      First decide whether each one tests outdated behaviour or a real bug.
- [ ] Flat light curve + noise → model offset within 0.1σ of the truth after
      the full pipeline.
- [ ] Slow trend (P = 10 d) is followed; a 2 h flare is not absorbed.

## Step 8: Savitzky-Golay passes

**Read:** `pipeline.py` 307–363, `savgoldetrending.py`, and
`altai.detrend_savgol`.

**Check:**
- Windows of 6 h and 3 h, applied after the baseline. In the synthetic runs
  they changed nothing in ~97 % of light curves. When do they matter, and
  can they clip long flares (FWHM of hours)?
- There are two `detrend_savgol` implementations (`altai.py` and
  `savgoldetrending.py`). Which one is used, and can the other one go?

**Tests** (`test_pipeline.py`):
- [ ] A flare with FWHM of 1 h keeps ≥ 90 % of its equivalent duration
      through the two Savitzky-Golay passes.

## Step 9: Gaussian-process model (`gp.py`, `pipeline._apply_gp_step`)

**Read:** `_identify_flare_mask` (32–103), `matchedfilter.py`
(`matched_filter_flare_mask` 250–384), `_segment_edge_mask` (106–129),
`_bin_training_per_segment` (132–231), `fit_gp_rotation` (234–541),
`_apply_gp_step` (389–502).

**Check:**
- **Training data.** The GP trains on the *original* flux minus the flare
  mask. The mask comes from the baseline stage plus the matched filter.
  Which flares slip through, and what does the one-sided clipping (3 rounds)
  then catch?
- ⚠ **Edge anchors override the flare mask.** A flare at a segment edge is
  forced into the training set, so the GP may absorb it.
- **Binning by 10.** Errors are propagated as sqrt(Σσ²)/n. Flares shorter
  than ~20 min are averaged into bins. Does that change what the clipping
  can see?
- **Kernel and bounds** (451–458):
  - P only within ±5 % of the Lomb-Scargle period.
  - Q0 and dQ within [1, 200].
  - Jitter within 0.05–2 × std(flux), which includes the variability.

  In injection-recovery, a parameter hit a bound in every light curve
  (period, jitter, dQ, f). Which bounds are physically motivated, and which
  hide a misfit?
- With 2–3 unrelated periods, `RotationTerm` at P and P/2 cannot fit them
  all. Should a non-periodic SHO term be added?
- **Offset** (484): `median(flux − gp)` over unmasked points. Is the median
  the right estimator if flare residue is present?

**Tests** (`test_gp.py`):
- [ ] `_identify_flare_mask` flags an injected flare plus its dilation, and
      does not mask a whole segment that has a constant offset.
- [ ] `_bin_training_per_segment` keeps both segment edges at full
      resolution and loses no points (already partly verified; turn it into
      a test).
- [ ] `fit_gp_rotation` on a quasi-periodic signal (P = 2 d) + flares → the
      model follows the modulation (residual scatter close to the noise) and
      not the flares (peak residual ≥ 90 % of the flare amplitude).
- [ ] A flare at a segment edge: document what happens (this test may show
      the ⚠ above).
- [ ] `return_std=True` returns the predictive standard deviation,
      `return_std=False` returns None.

## Step 10: Noise estimate (`noise.py`)

**Read:** `estimate_detrended_noise` (10–105).

**Check:**
- Rolling std over 100 cadences, after a one-sided 2.5σ clip and subtraction
  of the iterative median. Is the std biased by the clip (truncated
  distribution) or inflated by red noise?
- `find_flares` uses this σ for its detection threshold, so any bias here
  shifts the false-positive rate directly.

**Tests** (`test_noise.py`):
- [ ] White noise σ → estimate within 5 % of σ.
- [ ] White noise + flares → estimate within 10 % of σ.
- [ ] Quantity and plain-array input give the same result.

## Step 11: End to end

**Read:** `notebooks/injection_recovery.py` (`flare_finder_altai`) and
`altai.find_flares` / `measure_flare` as the consumers of the pipeline's
output.

**Check:**
- Do the recovery rates (S/N 2–5: 32 %, 5–10: 87 %, > 10: 100 %) and 2 false
  positives per 100 light curves match what you expect from the noise
  estimate and detection thresholds?
- Fast rotators lose faint flares (S/N 2–5: 22 % vs 35 %). Is that
  acceptable?

**Tests** (`test_end_to_end.py`, marked `slow`):
- [ ] A small injection-recovery run (5 light curves, fixed seeds):
      ≥ 95 % recovery for S/N > 10 and ≤ 1 false positive. This acts as a
      regression guard for everything above.
- [ ] One fast rotator (P = 9 h): no false positives.

---

## Afterwards

- [ ] Move the remaining items from `TODO.md` into the matching steps, or
      close them.
- [ ] Update the docstrings to the contract written down in step 1.
