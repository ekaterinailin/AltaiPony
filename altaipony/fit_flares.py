import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, medfilt, peak_prominences
from scipy.optimize import curve_fit
import emcee
import corner

from altaipony.fakeflares import flare_model_davenport2014

__all__ = [

    # Modeling
    "combined_model",         # baseline + flares
    "stacked_flare_model",    # flares-only
    "build_baseline",         # shared baseline evaluator

    # Optimization tools
    "model_selection",        # AIC/BIC scoring
    "log_likelihood",         # emcee likelihood
    "log_prior",              # emcee prior
    "log_posterior",          # emcee posterior

    # Light curve utils
    "extract_lc",             # crop a time segment
    "group_peaks",            # flare group detection

    # Core interface
    "fit_flares",             # top-level wrapper
    "fit_single_flare",       # inner fitting loop

    # Results and visualization
    "make_flare_table",       # result table
    "plot_flare_fit",         # individual fit plot
    "plot_all_fits",          # global LC + fit overlay
    "ed_from_model"           # equivalent duration calculator
]



def combined_model(t, *params):
    """
    Full flare model = polynomial baseline + stacked flares.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    *params : float
        Model parameters: first 5 are baseline coefficients [c0–c4],
        followed by flare parameters in groups of 3: [t_peak, fwhm, amp].

    Returns
    -------
    np.ndarray
        Total model flux = baseline + flare components.
    """
    # Validate input
    if not isinstance(t, np.ndarray):
        raise TypeError("Input t must be a NumPy array")
    if len(t) == 0:
        raise ValueError("Input t must not be empty")

    if len(params) < 5:
        raise ValueError("Must provide at least 5 parameters for baseline coefficients")

    baseline_coeffs = params[:5]
    flare_params = params[5:]

    if len(flare_params) % 3 != 0:
        raise ValueError("Flare parameters must be in groups of 3: [t_peak, fwhm, amp]")

    # Build model
    baseline = build_baseline(t, baseline_coeffs)
    flare_flux = stacked_flare_model(t, *flare_params) if flare_params else np.zeros_like(t)

    total_model = baseline + flare_flux

    return total_model


def stacked_flare_model(t, *params):
    """
    Flare-only model using a sum of Davenport et al. (2014) flares.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    *params : float
        Flattened list: [t_peak0, fwhm0, amp0, t_peak1, fwhm1, amp1, ...]

    Returns
    -------
    np.ndarray
        Flux from all stacked flares (no baseline).
    """
    if not isinstance(t, np.ndarray):
        raise TypeError("Time input must be a NumPy array.")
    if t.ndim != 1:
        raise ValueError("Time array must be 1D.")
    if len(t) == 0:
        raise ValueError("Time array must not be empty.")
    if len(params) % 3 != 0:
        raise ValueError("Flare parameters must come in groups of 3: [t_peak, fwhm, amp].")

    model = np.zeros_like(t)
    n_flares = len(params) // 3

    for i in range(n_flares):
        try:
            t_peak = params[3 * i]
            fwhm = params[3 * i + 1]
            amp = params[3 * i + 2]
            if not (np.isfinite(t_peak) and np.isfinite(fwhm) and np.isfinite(amp)):
                continue  # skip invalid flare
            model += flare_model_davenport2014(t, t_peak, fwhm, amp)
        except Exception as e:
            print(f"Warning: Failed to add flare {i}: {e}")
            continue

    return model



def build_baseline(time, coeffs):
    """
    Evaluate a 4th-order polynomial baseline at given times.

    Parameters
    ----------
    time : np.ndarray
        Time array [days].
    coeffs : list of float
        Baseline coefficients [c0, c1, c2, c3, c4].

    Returns
    -------
    np.ndarray
        Baseline flux values evaluated at `time`.
    """
    if not isinstance(time, np.ndarray):
        raise TypeError("Input 'time' must be a NumPy array.")
    if time.ndim != 1:
        raise ValueError("Time array must be 1D.")
    if len(time) == 0:
        raise ValueError("Time array must not be empty.")
    if len(coeffs) != 5:
        raise ValueError("Coefficient list must contain exactly 5 elements.")
    if not all(np.isfinite(coeffs)):
        raise ValueError("All baseline coefficients must be finite numbers.")

    t_centered = time - np.mean(time)
    c0, c1, c2, c3, c4 = coeffs

    baseline = c0 + c1*t_centered + c2*t_centered**2 + c3*t_centered**3 + c4*t_centered**4

    if baseline.shape != time.shape:
        raise RuntimeError("Output baseline shape mismatch.")

    return baseline



def log_likelihood(params, t, flux, flux_err):
    """
    Gaussian log-likelihood for the full model.

    Parameters
    ----------
    params : array-like
        Model parameters: [c0–c4] + [t_peak, fwhm, amp] × N.
    t : np.ndarray
        Time array.
    flux : np.ndarray
        Observed flux.
    flux_err : np.ndarray
        Observed errors.

    Returns
    -------
    float
        Log-likelihood value (higher is better).
    """
    if not (len(t) == len(flux) == len(flux_err)):
        raise ValueError("t, flux, and flux_err must have the same length")

    if len(t) == 0:
        raise ValueError("Input arrays must not be empty")

    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(flux)) or np.any(~np.isfinite(flux_err)):
        raise ValueError("Input arrays contain NaN or inf")

    model = combined_model(t, *params)

    if model.shape != flux.shape:
        raise ValueError("Model and flux must have the same shape")

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = ((flux - model) / flux_err) ** 2
        log_term = np.log(2 * np.pi * flux_err ** 2)
        chi2[~np.isfinite(chi2)] = 1e10  # large penalty
        log_term[~np.isfinite(log_term)] = 1e10

    return -0.5 * np.sum(chi2 + log_term)



def log_prior(params, t_bounds, amp_bounds, fwhm_bounds):
    """
    Flat prior: enforces bounds for flare parameters.

    Parameters
    ----------
    params : array-like
        Model parameters [c0–c4] + flares.
    t_bounds : list of float
        Time bounds for each t_peak (length = 2 × n_flares).
    amp_bounds : tuple of float
        (amp_min, amp_max)
    fwhm_bounds : tuple of float
        (fwhm_min, fwhm_max)

    Returns
    -------
    float
        0 if within bounds, -inf otherwise.
    """
    # Check types and lengths
    if len(params) < 5:
        raise ValueError("params must include at least 5 baseline coefficients")
    if (len(params) - 5) % 3 != 0:
        raise ValueError("flare parameters must be in groups of 3 after the first 5 baseline terms")

    n_flares = (len(params) - 5) // 3
    if len(t_bounds) != 2 * n_flares:
        raise ValueError("t_bounds must contain exactly 2 entries per flare")

    if not (np.isfinite(amp_bounds).all() and np.isfinite(fwhm_bounds).all()):
        raise ValueError("amp_bounds and fwhm_bounds must contain finite values")

    # Validate bounds
    for i in range(5, len(params), 3):
        t_peak = params[i]
        fwhm = params[i + 1]
        amp = params[i + 2]
        j = (i - 5) // 3

        if not (np.isfinite(t_peak) and np.isfinite(fwhm) and np.isfinite(amp)):
            return -np.inf

        if not (t_bounds[2*j] < t_peak < t_bounds[2*j + 1]):
            return -np.inf
        if not (fwhm_bounds[0] < fwhm < fwhm_bounds[1]):
            return -np.inf
        if not (amp_bounds[0] < amp < amp_bounds[1]):
            return -np.inf

    return 0.0



def log_posterior(params, t, flux, flux_err, t_bounds, amp_bounds, fwhm_bounds):
    """
    Posterior = log prior + log likelihood.

    Parameters
    ----------
    params : array-like
        Full model parameters.
    t : np.ndarray
        Time array.
    flux : np.ndarray
        Observed flux.
    flux_err : np.ndarray
        Flux uncertainties.
    t_bounds : list of float
        Bounds on each t_peak: [min0, max0, min1, max1, ...]
    amp_bounds : tuple
        (amp_min, amp_max)
    fwhm_bounds : tuple
        (fwhm_min, fwhm_max)

    Returns
    -------
    float
        Log posterior value.
    """
    if not (len(t) == len(flux) == len(flux_err)):
        raise ValueError("t, flux, and flux_err must all have the same length")
    if len(params) < 5:
        raise ValueError("params must include at least 5 baseline values")

    if np.any(~np.isfinite(params)):
        return -np.inf

    lp = log_prior(params, t_bounds, amp_bounds, fwhm_bounds)
    if not np.isfinite(lp):
        return -np.inf

    try:
        ll = log_likelihood(params, t, flux, flux_err)
    except Exception as e:
        print(f"Likelihood evaluation failed: {e}")
        return -np.inf

    return lp + ll



def model_selection(model, data, flux_err, params, method="bic"):
    """
    Compute AIC or BIC for a model fit.

    Parameters
    ----------
    model : np.ndarray
        Model flux.
    data : np.ndarray
        Observed flux.
    flux_err : np.ndarray
        Observational uncertainties.
    params : array-like
        List of fitted model parameters.
    method : str
        Either "aic" or "bic".

    Returns
    -------
    float
        Model score (lower is better).
    """
    if len(data) == 0 or len(model) == 0:
        return np.inf

    if not (len(data) == len(model) == len(flux_err)):
        raise ValueError("data, model, and flux_err must be the same length")

    if np.any(~np.isfinite(data)) or np.any(~np.isfinite(model)) or np.any(~np.isfinite(flux_err)):
        return np.inf

    k = len(params)
    n = len(data)

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = ((data - model) / flux_err) ** 2
        log_term = np.log(2 * np.pi * flux_err ** 2)
        chi2[~np.isfinite(chi2)] = 1e10
        log_term[~np.isfinite(log_term)] = 1e10

    loglike = -0.5 * np.sum(chi2 + log_term)

    if method == "bic":
        return k * np.log(n) - 2 * loglike
    elif method == "aic":
        return 2 * k - 2 * loglike
    else:
        raise ValueError("method must be 'aic' or 'bic'")


        
def extract_lc(time, flux, flux_err, t0, t1):
    """
    Extract a light curve segment between t0 and t1.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    flux : np.ndarray
        Corresponding flux array.
    flux_err : np.ndarray
        Corresponding flux error array.
    t0 : float
        Start time of the segment.
    t1 : float
        End time of the segment.

    Returns
    -------
    tuple of np.ndarray
        (time_window, flux_window, flux_err_window)
    """
    if not (len(time) == len(flux) == len(flux_err)):
        raise ValueError("time, flux, and flux_err must have the same length")

    if not np.isfinite(t0) or not np.isfinite(t1):
        raise ValueError("t0 and t1 must be finite numbers")

    if t0 >= t1:
        raise ValueError("t0 must be strictly less than t1")

    if len(time) == 0 or len(flux) == 0 or len(flux_err) == 0:
        raise ValueError("Input arrays must not be empty.")

    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(flux)) or np.any(~np.isfinite(flux_err)):
        raise ValueError("Input arrays must not contain NaNs or infs")

    mask = (time >= t0) & (time <= t1)

    if not np.any(mask):
        print(f"Warning: no points in window ({t0:.4f}, {t1:.4f})")
        return np.array([]), np.array([]), np.array([])

    return time[mask], flux[mask], flux_err[mask]



def group_peaks(tstarts, tstops, buffer=0.05):
    """
    Group nearby flares into temporal clusters.

    Parameters
    ----------
    tstarts : array-like
        Start times of individual flares.
    tstops : array-like
        Stop times of individual flares.
    buffer : float
        Time padding around each flare for merging.

    Returns
    -------
    List[List[int]]
        Groups of flare indices.
    """
    tstarts = np.asarray(tstarts, dtype=float)
    tstops = np.asarray(tstops, dtype=float)

    if tstarts.ndim != 1 or tstops.ndim != 1:
        raise ValueError("tstarts and tstops must be 1D arrays")

    if len(tstarts) != len(tstops):
        raise AssertionError("Mismatch in number of tstarts and tstops")

    if len(tstarts) == 0:
        return []

    if not np.isfinite(buffer) or buffer < 0:
        raise ValueError("buffer must be a non-negative finite float")

    if np.any(~np.isfinite(tstarts)) or np.any(~np.isfinite(tstops)):
        raise ValueError("tstarts and tstops must not contain NaN or inf")

    groups = []
    current = [0]

    for i in range(1, len(tstarts)):
        prev_end = tstops[current[-1]] + buffer
        curr_start = tstarts[i] - buffer

        if curr_start <= prev_end:
            current.append(i)
        else:
            groups.append(current)
            current = [i]

    groups.append(current)
    return groups



def fit_flares(time, flux, flux_err, tstarts, tstops,
               buffer=0.05, max_flares=3, method="curve_fit", plot=False, debug_plot=False):
    """
    Fit all detected flares in a light curve, including groups and individual members.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    flux : np.ndarray
        Flux values.
    flux_err : np.ndarray
        Flux uncertainties.
    tstarts, tstops : array-like
        Flare start/stop times.
    buffer : float
        Time buffer before/after each flare or group.
    max_flares : int
        Max flares to model per group.
    method : str
        "curve_fit" or "emcee"
    plot : bool
        Whether to show plots.
    debug_plot : bool
        Whether to show trial fits.

    Returns
    -------
    list of dict
        One result per group or group member.
    """
    # Input validation
    if not (len(time) == len(flux) == len(flux_err)):
        raise ValueError("Input arrays time, flux, and flux_err must have the same length.")

    if len(time) < 10:
        raise ValueError("Light curve too short to fit.")

    if len(tstarts) == 0 or len(tstops) == 0:
        raise ValueError("No flare intervals (tstarts, tstops) provided.")

    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(flux)) or np.any(~np.isfinite(flux_err)):
        raise ValueError("Input arrays contain NaN or inf.")
        
    if max_flares < 1:
        raise ValueError("max_flares must be >= 1")

    if method not in {"curve_fit", "emcee"}:
        raise ValueError("method must be 'curve_fit' or 'emcee'")
    
    results = []
    tstarts = np.asarray(tstarts)
    tstops = np.asarray(tstops)
    groups = group_peaks(tstarts, tstops, buffer=buffer)
    group_id = 1

    for group in groups:
        group_t0 = tstarts[group[0]] - buffer
        group_t1 = tstops[group[-1]] + buffer
        n_group = len(group)
        print(f"\n Fitting region from {group_t0:.5f} to {group_t1:.5f} "
              f"(max_flares = {max_flares * n_group})")
        t_win, f_win, fe_win = extract_lc(time, flux, flux_err, group_t0, group_t1)

        # Collect peaks + durations
        region_peaks, region_durations = [], []
        for idx in group:
            t0, t1 = tstarts[idx] - buffer, tstops[idx] + buffer
            t_seg, f_seg, _ = extract_lc(time, flux, flux_err, t0, t1)
            if len(t_seg) == 0 or len(f_seg) == 0:
                continue

            peaks, _ = find_peaks(f_seg, distance=2)

            if len(peaks) == 0:
                continue

            # Compute peak features
            t_peaks = t_seg[peaks]
            f_peaks = f_seg[peaks]
            proms = peak_prominences(f_seg, peaks)[0]
            widths_result = peak_widths(f_seg, peaks, rel_height=0.5)
            widths = widths_result[0] * np.median(np.diff(t_seg))

            # Keep only peaks within detected flare interval
            mask = (t_peaks >= tstarts[idx]) & (t_peaks <= tstops[idx])
            t_peaks = t_peaks[mask]
            f_peaks = f_peaks[mask]
            proms = proms[mask]
            widths = widths[mask]
            peaks = peaks[mask]

            if len(t_peaks) == 0:
                continue

            # Score = prominence × width (proxy for area / visibility)
            scores = proms * widths
            sort_idx = np.argsort(scores)[::-1]  # descending score
            N = min(max_flares, len(sort_idx))
            region_peaks.append(t_peaks[sort_idx[:N]])
            region_durations.append(widths[sort_idx[:N]])


            

        # Interleave peaks from all flares
        peaks, durations = [], []
        for i in range(max_flares):
            for r in range(len(region_peaks)):
                if i < len(region_peaks[r]):
                    peaks.append(region_peaks[r][i])
                    durations.append(region_durations[r][i])

        # Build guesses
        flare_guess_all, bounds_lower_all, bounds_upper_all = [], [], []
        for i in range(max_flares * len(group)):
            if i < len(peaks):
                tp = peaks[i]
                dur = durations[i]
            else:
                tp = 0.5 * (t_win[0] + t_win[-1])  # fallback midpoint
                dur = 0.01

            amp = np.max(f_win) - np.median(f_win)
            flare_guess_all += [tp, max(0.001, 0.33 * dur), amp]
            bounds_lower_all += [tp - 0.005, 0.001, 0.001]
            bounds_upper_all += [tp + 0.005, 0.1, 2 * amp]
            
            
        # Fit the full group
        result_group = fit_single_flare(
            t_win, f_win, fe_win,
            flare_guess_all, bounds_lower_all, bounds_upper_all,
            tstarts[group], tstops[group],
            method=method,
            max_flares=max_flares * len(group),
            debug_plot=debug_plot
        )

        if result_group is None:
            continue

        if len(group) > 1:
            result_group["fit_type"] = "group"
        elif result_group["n_flares"] > 1:
            result_group["fit_type"] = "group"
        else:
            result_group["fit_type"] = "single"

        result_group["group_index"] = group_id if len(group) > 1 or result_group["n_flares"] > 1 else ""
        result_group["t_range"] = (group_t0, group_t1)
        result_group["time"] = t_win
        result_group["flux"] = f_win
        result_group["flux_err"] = fe_win
        results.append(result_group)

        if plot:
            plot_flare_fit(t_win, f_win, result_group["model"], peaks, result_group["params"])

        # Add member-level results
        if len(group) > 1 or result_group["n_flares"] > 1:
            baseline = result_group["params"][:5]
            for i in range(result_group["n_flares"]):
                tp = result_group["t_peaks"][i]
                fwhm = result_group["fwhms"][i]
                amp = result_group["amplitudes"][i]
                flare_model = flare_model_davenport2014(t_win, tp, fwhm, amp)
                baseline_flux = build_baseline(t_win, baseline)
                model = baseline_flux + flare_model

                result_member = {
                    "params": baseline + [tp, fwhm, amp],
                    "model": model,
                    "t_peak": tp,
                    "fwhm": fwhm,
                    "amplitude": amp,
                    "fit_type": "group_member",
                    "group_index": group_id,
                    "t_range": (group_t0, group_t1),
                    "time": t_win,
                    "flux": f_win,
                    "flux_err": fe_win
                }
                results.append(result_member)

                if plot:
                    plot_flare_fit(t_win, f_win, model, [tp], result_member)

            group_id += 1

    return results


def fit_single_flare(time, flux, flux_err,
                     flare_guess_all, bounds_lower_all, bounds_upper_all,
                     tstarts, tstops,
                     method="curve_fit", max_flares=3,
                     fixed_baseline=None, debug_plot=False):
    """
    Fit a flare or flare group using a baseline + multi-flare model.

    Parameters
    ----------
    time : np.ndarray
        Time array for the region.
    flux : np.ndarray
        Flux values.
    flux_err : np.ndarray
        Uncertainties.
    flare_guess_all : list of float
        Flattened initial guesses for all flares.
    bounds_lower_all, bounds_upper_all : list of float
        Corresponding lower/upper bounds.
    tstarts, tstops : list of float
        Bounds for each flare (used in priors).
    method : str
        "curve_fit" or "emcee".
    max_flares : int
        Max number of flares to fit.
    fixed_baseline : list of float, optional
        Fixed polynomial baseline if provided.
    debug_plot : bool
        Show intermediate trial fits.

    Returns
    -------
    dict or None
        Best-fitting model dictionary or None on failure.
    """
    if len(time) == 0 or len(flux) == 0:
        return None  # skip empty region

    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(flux)) or np.any(~np.isfinite(flux_err)):
        print("Skipping fit: input contains NaN or inf.")
        return None

    if not (len(time) == len(flux) == len(flux_err)):
        print("Skipping fit: input arrays are mismatched in length.")
        return None

    best_result = None
    best_score = np.inf
    n_detected = max(1, max_flares)

    for n in range(1, n_detected + 1):
        flare_guess = flare_guess_all[:3 * n]
        bounds_lower = bounds_lower_all[:3 * n]
        bounds_upper = bounds_upper_all[:3 * n]

        if fixed_baseline is not None:
            baseline_guess = fixed_baseline
        else:
            baseline_guess = [np.median(flux), 0, 0, 0, 0]

        try:
            if method == "curve_fit":
                if fixed_baseline is not None:
                    def flare_only(t, *p): return combined_model(t, *(fixed_baseline + list(p)))
                    p0 = flare_guess
                    popt, _ = curve_fit(flare_only, time, flux, p0=p0,
                                        bounds=(bounds_lower, bounds_upper),
                                        sigma=flux_err, absolute_sigma=True, maxfev=10000)
                    full_params = fixed_baseline + list(popt)
                else:
                    p0 = baseline_guess + flare_guess
                    popt, _ = curve_fit(combined_model, time, flux, p0=p0,
                                        bounds=( [-np.inf]*5 + bounds_lower, [np.inf]*5 + bounds_upper ),
                                        sigma=flux_err, absolute_sigma=True, maxfev=10000)
                    full_params = list(popt)

                model = combined_model(time, *full_params)
                score = model_selection(model, flux, flux_err, full_params, method="bic")

                if debug_plot:
                    print(f"✓ Accepted model with n = {n}, BIC = {score:.1f}")
    
                t_peaks = [flare_guess_all[3*i] for i in range(n)]
                if debug_plot:
                    plot_flare_fit(time, flux, model, t_peaks, full_params,
                                   title=f"Trial Fit: n = {n} flares, BIC = {score:.1f}")

                if score < best_score:
                    best_result = {
                        "params": full_params,
                        "model": model,
                        "n_flares": n,
                        "score": score,
                        "t_peaks": [full_params[5 + 3*i] for i in range(n)],
                        "fwhms":  [full_params[5 + 3*i + 1] for i in range(n)],
                        "amplitudes": [full_params[5 + 3*i + 2] for i in range(n)]
                    }
                    best_score = score
                    
                if n == 1:
                    best_result["t_peak"] = full_params[5]
                    best_result["fwhm"] = full_params[6]
                    best_result["amplitude"] = full_params[7]


            elif method == "emcee":
                t_bounds = []
                for i in range(n):
                    tp = flare_guess_all[3 * i]
                    t_bounds += [tp - 0.005, tp + 0.005]

                amp_bounds = (0.001, np.max(flux))
                fwhm_bounds = (0.001, 0.1)

                if fixed_baseline is not None:
                    guess = flare_guess
                    ndim = len(guess)
                    pos = np.array(guess) + 1e-5 * np.random.randn(max(2 * ndim, 50), ndim)

                    def logpost_fixed(params, *args):
                        return log_posterior(fixed_baseline + list(params), *args)

                    sampler = emcee.EnsembleSampler(len(pos), ndim, logpost_fixed,
                                                    args=(time, flux, flux_err, t_bounds, amp_bounds, fwhm_bounds))
                else:
                    guess = baseline_guess + flare_guess
                    ndim = len(guess)
                    pos = np.array(guess) + 1e-5 * np.random.randn(max(2 * ndim, 50), ndim)

                    sampler = emcee.EnsembleSampler(len(pos), ndim, log_posterior,
                                                    args=(time, flux, flux_err, t_bounds, amp_bounds, fwhm_bounds))

                sampler.run_mcmc(pos, 3000, progress=True)
                samples = sampler.get_chain(discard=500, thin=10, flat=True)
                best_fit = np.median(samples, axis=0)

                full_params = fixed_baseline + list(best_fit) if fixed_baseline else list(best_fit)
                model = combined_model(time, *full_params)
                score = model_selection(model, flux, flux_err, full_params, method="bic")

                if debug_plot:
                    print(f"✓ Accepted model with n = {n}, BIC = {score:.1f}")

                t_peaks = [flare_guess_all[3*i] for i in range(n)]
                if debug_plot:  
                    plot_flare_fit(time, flux, model, t_peaks, full_params,
                                   title=f"Trial Fit: n = {n} flares, BIC = {score:.1f}")

                if score < best_score:
                    best_result = {
                        "params": full_params,
                        "model": model,
                        "n_flares": n,
                        "score": score,
                        "posterior_samples": samples,
                        "t_peaks": [full_params[5 + 3*i] for i in range(n)],
                        "fwhms":  [full_params[5 + 3*i + 1] for i in range(n)],
                        "amplitudes": [full_params[5 + 3*i + 2] for i in range(n)]
                    }
                    best_score = score

                    if n == 1:
                        best_result["t_peak"] = full_params[5]
                        best_result["fwhm"] = full_params[6]
                        best_result["amplitude"] = full_params[7]

        except Exception as e:
            print(f"{method} failed (n={n}): {e}")
            continue

    return best_result


def ed_from_model(time, model_flux, baseline_flux):
    """
    Compute equivalent duration (ED) in seconds.

    Parameters
    ----------
    time : np.ndarray
        Time array [days].
    model_flux : np.ndarray
        Total model flux (flare + baseline).
    baseline_flux : float or np.ndarray
        Baseline flux level (can be scalar or array).

    Returns
    -------
    float
        Equivalent duration in seconds.
    """
    if not (len(time) == len(model_flux)):
        raise ValueError("time and model_flux must have the same length")

    if isinstance(baseline_flux, np.ndarray) and len(baseline_flux) != len(time):
        raise ValueError("If baseline_flux is an array, it must match time/model_flux in length")

    if len(time) < 2:
        raise ValueError("time must contain at least 2 points")

    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(model_flux)):
        raise ValueError("Input arrays contain NaN or inf")

    if isinstance(baseline_flux, np.ndarray):
        if np.any(~np.isfinite(baseline_flux)):
            raise ValueError("baseline_flux contains NaN or inf")
    elif not np.isfinite(baseline_flux):
        raise ValueError("baseline_flux is not finite")

    time_sec = time * 86400  # convert from days to seconds
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = (model_flux - baseline_flux) / baseline_flux
        excess[~np.isfinite(excess)] = 0.0  # set NaNs/Infs to 0

    ed = np.sum(np.diff(time_sec) * excess[:-1])
    return ed




def make_flare_table(results, include_group_rows=False):
    """
    Build a flare summary table with uncertainties and EDs.

    Parameters
    ----------
    results : list of dict
        Output from fit_flares().
    include_group_rows : bool
        Whether to include group-level summary rows.

    Returns
    -------
    pd.DataFrame
        Table of flare parameters.
    """

    def extract_errors(samples, i, offset=5):
        try:
            t_arr = samples[:, offset + 3 * i]
            fwhm_arr = samples[:, offset + 3 * i + 1]
            amp_arr = samples[:, offset + 3 * i + 2]
            t_err = 0.5 * (np.percentile(t_arr, 84) - np.percentile(t_arr, 16))
            fwhm_err = 0.5 * (np.percentile(fwhm_arr, 84) - np.percentile(fwhm_arr, 16))
            amp_err = 0.5 * (np.percentile(amp_arr, 84) - np.percentile(amp_arr, 16))
            return t_err, fwhm_err, amp_err
        except Exception as e:
            print(f"Error extracting errors for flare {i}: {e}")
            return np.nan, np.nan, np.nan

    rows = []

    for res in results:
        fit_type = res.get("fit_type", "")
        group_index = str(res.get("group_index", ""))
        samples = res.get("posterior_samples", None)

        if fit_type == "group":
            if include_group_rows:
                rows.append({
                    "t_peak": "", "t_peak_err": "",
                    "fwhm": "", "fwhm_err": "",
                    "amplitude": "", "amplitude_err": "",
                    "ed_rec": "", "fit_type": "group",
                    "group_index": group_index,
                    "n_flares": res.get("n_flares", 1),
                })

            for i in range(res.get("n_flares", 0)):
                try:
                    t_peak = res["t_peaks"][i]
                    fwhm = res["fwhms"][i]
                    amp = res["amplitudes"][i]
                except (KeyError, IndexError):
                    t_peak = fwhm = amp = np.nan

                t_err, fwhm_err, amp_err = extract_errors(samples, i) if samples is not None else (np.nan, np.nan, np.nan)

                try:
                    flare_model = flare_model_davenport2014(res["time"], t_peak, fwhm, amp)
                    baseline = build_baseline(res["time"], res["params"][:5])
                    model = baseline + flare_model
                    ed_rec = ed_from_model(res["time"], model, baseline)
                except Exception:
                    ed_rec = np.nan

                rows.append({
                    "t_peak": t_peak, "t_peak_err": t_err,
                    "fwhm": fwhm, "fwhm_err": fwhm_err,
                    "amplitude": amp, "amplitude_err": amp_err,
                    "ed_rec": ed_rec, "fit_type": "group_member",
                    "group_index": group_index, "n_flares": 1
                })

        elif fit_type == "group_member":
            continue  # already added above

        else:  # single flare or fallback
            t_peak = res.get("t_peak", np.nan)
            fwhm = res.get("fwhm", np.nan)
            amp = res.get("amplitude", np.nan)
            baseline = build_baseline(res["time"], res["params"][:5])

            try:
                ed_rec = ed_from_model(res["time"], res["model"], baseline)
            except Exception:
                ed_rec = np.nan

            if samples is not None:
                try:
                    if samples.shape[1] >= 8:
                        t_err, fwhm_err, amp_err = extract_errors(samples, 0, offset=5)
                    elif samples.shape[1] == 3:
                        t_err, fwhm_err, amp_err = extract_errors(samples, 0, offset=0)
                    else:
                        t_err = fwhm_err = amp_err = np.nan
                except Exception:
                    t_err = fwhm_err = amp_err = np.nan
            else:
                t_err = fwhm_err = amp_err = np.nan

            rows.append({
                "t_peak": t_peak, "t_peak_err": t_err,
                "fwhm": fwhm, "fwhm_err": fwhm_err,
                "amplitude": amp, "amplitude_err": amp_err,
                "ed_rec": ed_rec, "fit_type": fit_type,
                "group_index": group_index,
                "n_flares": res.get("n_flares", 1)
            })

    return pd.DataFrame(rows)



def plot_flare_fit(time, flux, model, t_peaks, params, residuals=True, title=None):
    """
    Plot the flare fit with optional residuals.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    flux : np.ndarray
        Observed flux.
    model : np.ndarray
        Fitted model flux.
    t_peaks : list of float
        Peak times for annotation.
    params : dict or list
        Fitted parameters. Can be dict with 'params' or a flat list.
    residuals : bool
        Whether to show residuals subplot.
    title : str or None
        Optional plot title.
    """
    if isinstance(params, dict):
        params = params["params"]

    baseline = build_baseline(time, params[:5])
    residual = flux - model

    plt.figure(figsize=(10, 5))

    # Top: flux + model
    plt.subplot(211)
    plt.xlim(time[0], time[-1])
    plt.plot(time, flux, label="Flux [e⁻/s]", lw=1)
    plt.plot(time, model, label="Model", lw=2, linestyle="--")
    plt.plot(time, baseline, label="Baseline", lw=1.5, linestyle="dotted", color="lime")
    for tp in t_peaks:
        plt.axvline(tp, color="r", linestyle=":", lw=0.8)
    plt.xlabel("Time [days]")
    plt.ylabel("Flux [e⁻/s]")
    plt.legend()
    if title:
        plt.title(title)
    else:
        plt.title("Flare Fit")

    # Bottom: residuals
    if residuals:
        plt.subplot(212)
        plt.xlim(time[0], time[-1])
        plt.plot(time, residual, label="Residual", lw=1)
        plt.axhline(0, color="gray", linestyle="--")
        plt.xlabel("Time [days]")
        plt.ylabel("Residual")
        plt.legend()

    plt.tight_layout()
    plt.show()

    
def plot_all_fits(time, flux, results):
    """
    Plot the full light curve with overlaid flare model fits.

    Parameters
    ----------
    time : np.ndarray
        Full light curve time array.
    flux : np.ndarray
        Full light curve flux array.
    results : list of dict
        List of result dictionaries from `fit_flares`.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(time, flux, label='Observed Flux', lw=1)

    # Global baseline fit
    if len(time) > 15000:
        time_sub = time[::10]
        flux_sub = medfilt(flux, kernel_size=5)[::10]
    else:
        time_sub = time
        flux_sub = flux

    t0 = np.mean(time_sub)
    flux_median = np.median(flux_sub)
    t_centered = time_sub - t0
    coeffs = np.polyfit(t_centered, flux_sub - flux_median, deg=4)
    baseline_poly = np.poly1d(coeffs)
    baseline_full = baseline_poly(time - t0) + flux_median

    # Plot the baseline!
    plt.plot(time, baseline_full, color='lime', linestyle='--', lw=1.5, label="Global Baseline")

    # Avoid redundant labels
    seen_labels = set()

    for res in results:
        label = f"{res['fit_type']} (n={res.get('n_flares', 1)})"
        label = label if label not in seen_labels else None
        seen_labels.add(label)

        plt.plot(res["time"], res["model"], lw=1.5, alpha=0.6, label=label)

    plt.xlabel("Time [days]")
    plt.xlim(time[0], time[-1])
    plt.ylabel("Flux [e⁻/s]")
    plt.title("Flare Fits Over Full Light Curve")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()
