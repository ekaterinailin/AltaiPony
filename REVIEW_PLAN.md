# Review plan: `altaipony/detrending`

A step-by-step read-through of the detrending code, ordered along the data
flow and organised around one question per step: *is the science sound?*
Every step ends with tests that pin down the behaviour you have convinced
yourself of. The tests go into `altaipony/tests/detrending/`.

Line numbers were last refreshed on 2026-10-12, after the 2-day minimum
damping time and the residual-period term were added to the GP (the commit
after `f7a0351`).

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
    `baseline=` and keeps amplitudes relative. Mode amplitudes are
    log-uniform in 0.01–5 %, and 20 % of stars are spotless
    (`spotless_fraction`), so weakly and non-variable stars are covered.)
  - Also provide a `FlareLightCurve` fixture built from it.

  Done: `make_lc(duration, cadence_min, gaps, modes, flares, noise, scale,
  seed)` returns `(lc, flare_table)` and is available as a
  session fixture `make_lc`; `synthetic_lc` is the default light curve
  (8 days, e-/s at 3400, spots, a gap, two flares). `test_make_lc.py`
  checks it.
- [x] Register a `slow` marker (done in `conftest.py`) for tests
      that run the full pipeline. Keep the other tests short: a few thousand
      cadences, a few seconds each.
- [x] Run command, memory-capped:
  ```bash
  MPLBACKEND=Agg systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
      choom -n 1000 -- python -m pytest altaipony/tests/detrending -m "not slow"
  ```
  or, with a watchdog that also stops the job when the machine runs low on
  memory, `scripts/guarded_run.sh` (see its header for options):
  ```bash
  scripts/guarded_run.sh -- python -m pytest altaipony/tests/detrending -m "not slow"
  cd notebooks && ../scripts/guarded_run.sh -w 2000 -l injrec.log -- python -u injection_recovery.py
  ```

## Step 1: The pipeline's contract

**Read:** `pipeline.py`, `custom_detrending` (32–239) top to bottom, without
going into the helpers yet; then `_apply_gp_step` (301–457).

**Check:**
- What exactly does `custom_detrending` promise to return? Write it down in
  one sentence. It should cover what `detrended_flux` means, its zero point,
  its units, and which other attributes (`it_med`, `gp_model`) are set.
- ✓ **Zero point depends on the path** (resolved). It used to be 1.0 with
  the GP and the input flux level without it. Now both paths return relative
  flux at 1.0: the GP path as `1 + (flux − model) / level` (440–448), the
  non-GP path divided by its global iterative median (223–229).
- ✓ **Units and scale** (resolved). The GP path used to return
  "1 + residual in the flux's own units", so a 1 % flare in 3400 e-/s data
  came out as 34. The residual is now divided by the quiescent level, and
  the injection-recovery light curves are in e-/s (1000–100000) to keep this
  tested. Still open: `detrended_flux` is relative but labelled e-/s (229,
  448; see `TODO.md`).
- Which steps only produce *masks*, and which produce the *output*? In the GP
  path, the baseline result only feeds the GP's flare mask.

**Done so far** (`tests/detrending/test_pipeline.py`): baseline at exactly
1 and scale invariance for both the GP and the non-GP path; relative
amplitudes from `find_flares`. Fixed along the way: normalisation of both
paths, GP fitted on normalised flux, `flux_err` restored before the GP.

**Tests** (`test_pipeline.py`):
- [ ] Output has the same length and time stamps as the input, keeps NaNs
      where the input had them, and sets `it_med` and `gp_model`.
- [x] Zero point: a flat, flare-free stretch of `detrended_flux` sits at the
      documented level, **for both** the GP path (every star by default,
      with or without spots) and the non-GP path (`use_gp=False`).
- [x] **Scale invariance:** the same light curve with `scale=1` and
      `scale=3400` gives the same *relative* flare amplitude and the same
      equivalent duration. I expect this to fail today if the ⚠ above is real.
- [ ] Fix the 4 Quantity failures in `test_pipeline.py::TestCustomDetrending`
      (`test_measure_flare_*`, `test_custom_detrending_removes_variability`)
      once you have decided what type `detrended_flux` should have.

## Step 2: Gaps and cadence

**Read:** `pipeline.py` 158–165.

**Check:**
- ⚠ `dt = np.mean(np.diff(time))` (158) includes the gaps. A single 1-day
  gap in 25 days of 2-min data raises the mean step noticeably. That inflates
  `maxgap * dt`, the gap threshold. Should this be the median?
- `interpolate_missing_cadences` fills missing cadences inside continuous
  windows (flagged `interpolated`). They are part of the baseline and GP
  fits and dropped from the output at the end. Should the fits ignore them
  too?

**Tests** (`test_pipeline.py`):
- [ ] A light curve with one long gap is split into exactly two segments.
- [ ] Cadences removed from inside a window are filled for the fits but are
      not in the output, and no interpolated cadence ends up in a flare.

## Step 3: Detecting the rotation period

**Read:** `periodicity.py` (`lomb_scargle` 8–57,
`detect_strong_periodicity` 60–124); call site `pipeline.py` 175–191.

**Settled by injection-recovery (2026-10-09 to 10-12):**
- The periodogram no longer decides whether the GP runs: the GP runs for
  every star and only takes its starting period from the highest peak.
  Gating on a semi-amplitude (> 0.1 %) or S/N (> 3) threshold excluded stars
  the GP helps (S/N > 3: 11 flares lost); even on spotless stars the GP
  recovers more flares than the polynomial baseline alone (133 vs. 119 of
  164, no false positives).
- The Baluev FAP is tiny for almost everything: all 100 synthetic stars,
  including spots of 0.01 % far below the per-cadence noise, and flares
  alone on a spotless star (FAP ~1e-60, at roughly the spacing of the
  flares). The periodogram is therefore computed without the baseline's
  flare mask, which lowers that to ~1e-33 but does not remove it (the
  decay tails are not masked). The FAP threshold now only gates the
  second-period search.

**Check:**
- Search range is 0.1 d to half the light curve. What happens with periods
  near the ends of that range?
- On a spotless star the starting period comes from flares or noise. The GP
  then settles at a small amplitude; is its period bound (±5 %) ever a
  problem there?

**Tests** (`test_periodicity.py`):
- [ ] A pure sine with P ∈ {0.3, 2, 8} d is recovered to within 1 %.
- [ ] White noise only → not significant (FAP > 1e-3).
- [ ] Light curve shorter than 2 × `period_min` → not periodic, no error.
- [ ] `lomb_scargle` drops non-finite points and never searches periods
      longer than the data span.

## Step 4: Choosing the baseline (removed)

The polynomial baseline is now the only one. An injection-recovery
comparison (seed 92, 100 LCs) gave about the same recovery for all four
choices (662–668 of 787) but very different false positives: polynomial 7,
`auto` 68, spline 115, multi-sine 60. The other three left the GP's starting
flare mask to `_identify_flare_mask` on the baseline residual, which masked
the variability maxima the baseline could not follow (e.g. 15.6 % of LC 57),
so the GP never fitted them and left hours-long bumps. The multi-sine and
spline baselines, `baseline_method`, and `_identify_flare_mask` were removed.

## Step 5: Polynomial baseline (`baselines/polynomial.py`)

The largest module. Review it in five parts.

**5a Preliminary flare mask:** `preliminary_flare_mask` (166–220),
`compute_rolling_local_sigma` (126–164).
- Is a 5σ cut on a rough 2-day polynomial residual conservative enough that
  real variability is never masked as flares?

**5b Window fits and blending:** `fit_window_grid` (223–339),
`run_all_windows` (342–401), `score_segment` (440–483),
`assemble_combined_trend` (486–580).
- Degree-4 polynomials in 0.4/0.6/0.8 d windows, on normal and half-shifted
  grids. Which variability timescales can they follow, and which flare
  durations would they eat into? A flare with FWHM ~1 h is a sizeable
  fraction of a 0.4 d window.
- Edge points are up-weighted 8× (`n_edge`, `edge_weight`). Why?
- `score_segment` mixes R²_adj, |log χ²_red|, masked fraction and point
  count with weights 0.15 / 0.30 / 0.05. The weights are heuristic. The
  softmax over scores then decides how sharply one fit wins. Is that
  sensible?

**5c Second pass:** in `run_detrending` (766–949), the 3σ mask from the
first-pass residual and the refit.

**5d Rotation correction (6–36 h): removed.** It mattered when the polynomial
baseline's output was the final detrended flux (48 false positives without
it then). Since the GP runs for every star, the baseline only supplies the
GP's starting flare mask, and injection-recovery (100 LCs) was identical
cadence for cadence with and without the correction.

**5e Centring and final mask:** `center_residual_by_moving_average`
(609–656), `build_safe_final_flare_mask` (659–764).
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

## Step 6: Multi-sine baseline (removed, see step 4)

## Step 7: Spline baseline (removed, see step 4)

## Step 8: Savitzky-Golay passes (removed)

The two passes (6 h, then 3 h) were removed from `custom_detrending`. On
the GP path they no longer affected anything (the GP is trained on the
original flux with the baseline's flare mask); injection-recovery with both,
either or neither pass gave identical results for all 90 GP stars of a
sample with amplitudes down to 0.01 %. For the 10 stars without a GP, no
filter recovered 44 of 57 flares vs. 42 with both, and no setting had false
positives. Not covered by that test: short-timescale systematics in real
data (scattered light, momentum dumps), which the polynomial windows
(0.4–0.8 d) may not follow. `FlareLightCurve.detrend("savgol")` and its two
`detrend_savgol` implementations (`altai.py`, `savgoldetrending.py`) still
exist outside the detrending package.

## Step 9: Gaussian-process model (`gp.py`, `pipeline._apply_gp_step`)

**Read:** `matchedfilter.py`
(`matched_filter_flare_mask` 250–384), `_segment_edge_mask` (31–54),
`_bin_training_per_segment` (57–156), `fit_gp_rotation` (159–488);
in `pipeline.py`, `_apply_gp_step` (301–457) with its fits (353–433:
first fit, second-period search, residual-period term, matched-filter
refit), `_second_period_candidates` (242–279) and `_residual_period`
(282–298).

**Check:**
- **Training data.** The GP trains on the *original* flux minus the flare
  mask. The mask comes from the baseline stage; the matched filter then runs
  on the residual of a first GP fit, and the GP is refitted with its
  detections added. Which flares slip through, and what does the one-sided
  clipping (3 rounds) then catch?
  (Before, the matched filter ran on the baseline residual and masked every
  rotation maximum the baseline could not follow: 13–25 % of the light curve
  for fast, strongly modulated stars, e.g. seed 154 = LC 62, which the GP
  then never saw. Moving it after a first GP fit took injection-recovery from
  654/787 recovered and 74 false positives to 667/787 and 20.)
- ⚠ **Edge anchors override the flare mask.** A flare at a segment edge is
  forced into the training set, so the GP may absorb it.
- **Binning by 10.** Errors are propagated as sqrt(Σσ²)/n. Flares shorter
  than ~20 min are averaged into bins. Does that change what the clipping
  can see?
- **Kernel and bounds** (`gp.py` 378–423):
  - P only within ±5 % of the Lomb-Scargle period.
  - Q0 at least `2π · τ_min / P − 1/2` (and ≥ 1), with a minimum damping
    time τ_min = 2 d (`min_damping_days`): each term keeps its shape for at
    least 2 days. dQ within [1, 200].
  - Jitter within 0.05–2 × std(flux), which includes the variability.

  A check of the 100 injection-recovery stars before τ_min (2026-10-12):
  Q0 sat at its lower bound of 1 in 49 stars (all 22 spotless), the period
  at its ±5 % bound in 42, the jitter at its lower bound in 50 (the script's
  error bars, 0.1 % of the flux, already exceed the true noise for most
  stars). The P/2 term's damping time had a median of 2.4 d, 10 % below
  0.7 d. τ_min = 2 d changed recovery by at most 2 flares (660 vs. 658)
  but halved the false positives (1 vs. 2), so it was kept; τ_min = 1 and
  5 d were no better. Which of the remaining bounds hide a misfit?
- **Residual-period term** (`gp_residual_term`, `_residual_period`): after
  the fit and the second-period search, the periodogram of the GP residual
  (without the flare mask) is checked; if its highest peak has FAP below
  `periodicity_fap_threshold`, the GP is refitted with an extra rotation
  term at that period, kept if it lowers the residual scatter. It catches
  what the kernel cannot follow, typically a harmonic the terms do not
  cover (LC 21 and 70: 0.346 d, half of a 0.69 d mode). Injection-recovery:
  661/787 (+1), 0 false positives (from 1); 4 light curves change.
  - Like the second-period search, it uses the scatter check, not a
    likelihood comparison. Can it ever add a term that absorbs flares, e.g.
    on a star with many flares at roughly regular spacing?
- With 2–3 unrelated periods, `RotationTerm` at P and P/2 cannot fit them
  all. Tried (seed 92, 100 LCs, two-pass GP as reference: 667/787
  recovered, 20 false positives):
  - a non-periodic SHO term (Q = 1/√2, timescale ≥ 0.5 d): 670 and 18. It
    cannot follow a second *fast* mode, the actual problem (e.g. LC 21:
    GP locks onto 2.8 d, the 0.69 d mode is left over), and made LC 62
    worse. Removed.
  - `gp_second_rotation=True`: a second `RotationTerm` at the best of up to
    three candidate periods from the flux periodogram (excluding P and P/2,
    same FAP threshold), kept only if it lowers the residual
    scatter: 668 and 7, no flare lost, ~20 % more runtime. LC 21 and 70 go
    from 869 / 1399 ppm residual scatter to 401 / 397 ppm (noise 327 / 329).
    Candidates are needed because the strongest remaining peak can be a
    sidelobe of the rotation signal from its changing amplitude (LC 70:
    3.7 d). Now the default.
  - ⚠ **Short light curves.** On 8–12-day two-period stars with changing
    amplitudes, the candidates are often sidelobes of the first period that
    lie just outside the ±10 % exclusion (frequency resolution 1/T is
    coarse), so the second term misses the second mode or picks a sidelobe.
    The scatter check prevents a worse overall fit, but in one case the
    number of > 5σ residuals still rose (129 → 222). On 25-day light curves
    it works reliably. Should the exclusion width scale with 1/T?
  - Should the residual-scatter criterion be a likelihood/BIC comparison
    instead?
- **Offset** (`pipeline.py` 442): `median(flux − gp)` over unmasked
  points. Is the median the right estimator if flare residue is present?

**Tests** (`test_gp.py`):
- [x] A 25-day star with two unrelated spot modes (3 d and 0.69 d, changing
      amplitudes): one rotation term leaves > 3x the white noise, the second
      term brings the residual within 1.5x.
- [ ] A fast, strongly modulated rotator (P ≈ 6 h, amplitude rising from 0
      to ~12 % over a few days, as in seed 154 of `generate_synthetic_lc`)
      leaves residual scatter within ~2× the white noise and no spurious
      flares at the rotation maxima.
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
- [ ] A star whose GP residual keeps a periodic signal at a harmonic the
      terms do not cover gets a residual-period term, and its residual
      scatter drops; a star with only white noise and flares gets none.

## Step 10: Noise estimate (`noise.py`)

**Read:** `estimate_detrended_noise` (10–105).

**Check:**
- Rolling std over 100 cadences, after a one-sided 2.5σ clip and subtraction
  of the iterative median. Is the std biased by the clip (truncated
  distribution) or inflated by red noise?
- `find_flares` uses this σ for its detection threshold, so any bias here
  shifts the false-positive rate directly.

**Tests** (`test_noise.py`, which already checks σ = 40 within ±5, with and
without a flare, and with NaNs):
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
