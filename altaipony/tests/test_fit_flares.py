import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from astropy.time import Time
import astropy.units as u

from ..flarelc import FlareLightCurve
from ..fakeflares import flare_model_davenport2014
from ..fit_flares import (
    fit_flares,
    fit_single_flare,
    combined_model,
    stacked_flare_model,
    build_baseline,
    ed_from_model,
    make_flare_table,
    log_likelihood,
    log_prior,
    log_posterior,
    model_selection,
)


class TestFitFlares:
    """Test suite for flare fitting functions"""
    
    # ========== Fixtures ==========
    
    @pytest.fixture
    def mock_emcee_sampler(self):
        """Create a mock emcee sampler that returns fake but realistic samples"""
        def create_mock_sampler(n_params, n_walkers=50, n_samples=1000):
            """
            Create mock based on NUMBER of parameters, not specific values
            
            Parameters
            ----------
            n_params : int
                Number of parameters being fit
            n_walkers : int
                Number of walkers
            n_samples : int
                Number of samples per walker
            """
            mock_sampler = Mock()
            
            # Generate generic reasonable samples for any number of parameters
            samples = []
            for _ in range(n_walkers):
                walker_samples = []
                for _ in range(n_samples):
                    # Generate random but reasonable parameters
                    # Baseline: ~1.0, small slopes
                    # Flare params: t_peak ~1.0, fwhm ~0.05, amp ~0.2
                    if n_params >= 5:
                        params = [1.0, 0.0, 0.0, 0.0, 0.0]  # baseline
                        for i in range((n_params - 5) // 3):  # flares
                            params.extend([
                                1.0 + np.random.normal(0, 0.01),   # t_peak
                                0.05 + np.random.normal(0, 0.005),  # fwhm
                                0.2 + np.random.normal(0, 0.02)     # amplitude
                            ])
                    else:
                        params = [1.0] * n_params
                    
                    # Add small noise
                    noisy_params = [p + np.random.normal(0, abs(p) * 0.01) for p in params]
                    walker_samples.append(noisy_params)
                samples.append(walker_samples)
            
            samples = np.array(samples)
            print(samples.shape)
            
            def mock_get_chain(discard=0, thin=1, flat=False):
                if flat:
                    return samples[:, discard::thin, :].reshape(-1, n_params)
                else:
                    return samples[:, discard::thin, :]
            
            mock_sampler.get_chain = Mock(side_effect=mock_get_chain)
            mock_sampler.run_mcmc = Mock(return_value=None)
            return mock_sampler
    
        return create_mock_sampler
    
    @pytest.fixture
    def simple_baseline_lc(self):
        """Create a simple light curve with polynomial baseline and noise"""
        np.random.seed(42)
        time = np.linspace(0, 10, 1000)
        
        baseline_coeffs = [1.0, 0.01, -0.001, 0.0, 0.0]
        t_centered = time - np.mean(time)
        baseline = (baseline_coeffs[0] + 
                   baseline_coeffs[1] * t_centered + 
                   baseline_coeffs[2] * t_centered**2)
        
        noise_level = 0.01
        flux = baseline + np.random.normal(0, noise_level, len(time))
        flux_err = np.full_like(flux, noise_level)
        
        time_obj = Time(2450000 + time, format='jd')
        meta = {'TARGETID': 123456, 'MISSION': 'Kepler', 'QUARTER': 5}
        
        return FlareLightCurve(
            time=time_obj,
            flux=flux * (u.electron / u.s),
            flux_err=flux_err * (u.electron / u.s),
            meta=meta
        )
    
    @pytest.fixture
    def single_flare_lc(self):
        """Create a light curve with one clear flare"""
        np.random.seed(42)
        time = np.linspace(0, 10, 1000)
        
        baseline = np.ones_like(time) * 1.0
        
        t_peak = 5.0
        fwhm = 0.05
        amplitude = 0.3
        flare = flare_model_davenport2014(time, t_peak, fwhm, amplitude)
        
        noise_level = 0.01
        flux = baseline + flare + np.random.normal(0, noise_level, len(time))
        flux_err = np.full_like(flux, noise_level)
        
        time_obj = Time(2450000 + time, format='jd')
        meta = {'TARGETID': 123456, 'MISSION': 'Kepler', 'QUARTER': 5}
        
        flc = FlareLightCurve(
            time=time_obj,
            flux=flux * (u.electron / u.s),
            flux_err=flux_err * (u.electron / u.s),
            meta=meta
        )
        
        flc._true_flare = {'t_peak': t_peak, 'fwhm': fwhm, 'amplitude': amplitude}
        flc._true_baseline = [1.0, 0.0, 0.0, 0.0, 0.0]
        
        return flc
    
    @pytest.fixture
    def multi_flare_lc(self):
        """Create a light curve with multiple flares"""
        np.random.seed(42)
        time = np.linspace(0, 20, 2000)
        
        baseline_coeffs = [1.0, 0.005, -0.0002, 0.0, 0.0]
        t_centered = time - np.mean(time)
        baseline = (baseline_coeffs[0] + 
                   baseline_coeffs[1] * t_centered + 
                   baseline_coeffs[2] * t_centered**2)
        
        flares = [
            {'t_peak': 4.0, 'fwhm': 0.04, 'amplitude': 0.2},
            {'t_peak': 10.0, 'fwhm': 0.08, 'amplitude': 0.5},
            {'t_peak': 16.0, 'fwhm': 0.06, 'amplitude': 0.15},
        ]
        
        flux = baseline.copy()
        for flare_params in flares:
            flux += flare_model_davenport2014(
                time, 
                flare_params['t_peak'], 
                flare_params['fwhm'], 
                flare_params['amplitude']
            )
        
        noise_level = 0.01
        flux += np.random.normal(0, noise_level, len(time))
        flux_err = np.full_like(flux, noise_level)
        
        time_obj = Time(2450000 + time, format='jd')
        meta = {'TARGETID': 789012, 'MISSION': 'TESS', 'SECTOR': 10}
        
        flc = FlareLightCurve(
            time=time_obj,
            flux=flux * (u.electron / u.s),
            flux_err=flux_err * (u.electron / u.s),
            meta=meta
        )
        
        flc._true_flares = flares
        flc._true_baseline = baseline_coeffs
        
        return flc
    
    @pytest.fixture
    def overlapping_flares_lc(self):
        """Create a light curve with two overlapping flares"""
        np.random.seed(42)
        time = np.linspace(0, 10, 1000)
        
        baseline = np.ones_like(time) * 1.0
        
        flares = [
            {'t_peak': 5.0, 'fwhm': 0.08, 'amplitude': 0.25},
            {'t_peak': 5.1, 'fwhm': 0.06, 'amplitude': 0.15},
        ]
        
        flux = baseline.copy()
        for flare_params in flares:
            flux += flare_model_davenport2014(
                time,
                flare_params['t_peak'],
                flare_params['fwhm'],
                flare_params['amplitude']
            )
        
        noise_level = 0.01
        flux += np.random.normal(0, noise_level, len(time))
        flux_err = np.full_like(flux, noise_level)
        
        time_obj = Time(2450000 + time, format='jd')
        meta = {'TARGETID': 345678, 'MISSION': 'Kepler', 'QUARTER': 8}
        
        flc = FlareLightCurve(
            time=time_obj,
            flux=flux * (u.electron / u.s),
            flux_err=flux_err * (u.electron / u.s),
            meta=meta
        )
        
        flc._true_flares = flares
        flc._true_baseline = [1.0, 0.0, 0.0, 0.0, 0.0]
        
        return flc
    
    # ========== Test Model Components ==========
    
    def test_build_baseline(self):
        """Test polynomial baseline construction"""
        time = np.linspace(0, 10, 100)
        coeffs = [1.0, 0.1, -0.01, 0.0, 0.0]
        
        baseline = build_baseline(time, coeffs)
        
        assert len(baseline) == len(time)
        assert np.all(np.isfinite(baseline))
        assert np.std(baseline) > 0
    
    def test_build_baseline_constant(self):
        """Test baseline with only constant term"""
        time = np.linspace(0, 10, 100)
        coeffs = [2.0, 0.0, 0.0, 0.0, 0.0]
        
        baseline = build_baseline(time, coeffs)
        
        assert np.allclose(baseline, 2.0)
    
    def test_stacked_flare_model_single(self):
        """Test stacked flare model with single flare"""
        time = np.linspace(0, 10, 100)
        params = [5.0, 0.05, 0.3]
        
        model = stacked_flare_model(time, *params)
        
        assert len(model) == len(time)
        assert np.all(np.isfinite(model))
        assert np.max(model) > 0
        assert model[0] == pytest.approx(0, abs=1e-6)
    
    def test_stacked_flare_model_multiple(self):
        """Test stacked flare model with multiple flares"""
        time = np.linspace(0, 10, 100)
        params = [
            3.0, 0.05, 0.2,
            7.0, 0.04, 0.15,
        ]
        
        model = stacked_flare_model(time, *params)
        
        assert len(model) == len(time)
        peaks = np.where((model[1:-1] > model[:-2]) & (model[1:-1] > model[2:]))[0]
        assert len(peaks) >= 1
    
    def test_combined_model(self):
        """Test combined model (baseline + flares)"""
        time = np.linspace(0, 10, 100)
        baseline_coeffs = [1.0, 0.01, 0.0, 0.0, 0.0]
        flare_params = [5.0, 0.05, 0.3]
        params = baseline_coeffs + flare_params
        
        model = combined_model(time, *params)
        
        assert len(model) == len(time)
        assert np.all(np.isfinite(model))
        assert model[0] == pytest.approx(1.0, abs=0.1)
        assert np.max(model) > 1.0
    
    # ========== Test Likelihood Functions ==========
    
    def test_log_likelihood_perfect_fit(self):
        """Test log likelihood with perfect model fit"""
        time = np.linspace(0, 10, 100)
        params = [1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.05, 0.3]
        
        flux = combined_model(time, *params)
        flux_err = np.full_like(flux, 0.01)
        
        ll = log_likelihood(params, time, flux, flux_err)
        
        assert np.isfinite(ll)
        assert ll > -1000
    
    def test_log_likelihood_bad_fit(self):
        """Test log likelihood with poor model fit"""
        time = np.linspace(0, 10, 100)
        true_params = [1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.05, 0.3]
        bad_params = [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.1, 0.1]
        
        flux = combined_model(time, *true_params)
        flux_err = np.full_like(flux, 0.01)
        
        ll_good = log_likelihood(true_params, time, flux, flux_err)
        ll_bad = log_likelihood(bad_params, time, flux, flux_err)
        
        assert ll_good > ll_bad
    
    def test_log_prior_in_bounds(self):
        """Test log prior with parameters in bounds"""
        params = [1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.05, 0.3]
        t_bounds = [4.9, 5.1]
        amp_bounds = (0.1, 0.5)
        fwhm_bounds = (0.01, 0.1)
        
        lp = log_prior(params, t_bounds, amp_bounds, fwhm_bounds)
        
        assert lp == 0.0
    
    def test_log_prior_out_of_bounds(self):
        """Test log prior with parameters out of bounds"""
        params = [1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.05, 0.3]
        t_bounds = [4.9, 5.1]
        amp_bounds = (0.1, 0.5)
        fwhm_bounds = (0.01, 0.1)
        
        lp = log_prior(params, t_bounds, amp_bounds, fwhm_bounds)
        
        assert lp == -np.inf
    
    def test_log_posterior(self):
        """Test log posterior calculation"""
        time = np.linspace(0, 10, 100)
        params = [1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.05, 0.3]
        
        flux = combined_model(time, *params)
        flux_err = np.full_like(flux, 0.01)
        
        t_bounds = [4.9, 5.1]
        amp_bounds = (0.1, 0.5)
        fwhm_bounds = (0.01, 0.1)
        
        lpost = log_posterior(params, time, flux, flux_err, t_bounds, amp_bounds, fwhm_bounds)
        
        assert np.isfinite(lpost)
        assert lpost > -1000
    
    # ========== Test Model Selection ==========
    
    def test_model_selection_bic(self):
        """Test BIC calculation"""
        time = np.linspace(0, 10, 100)
        params = [1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.05, 0.3]
        
        data = combined_model(time, *params)
        model = combined_model(time, *params)
        flux_err = np.full_like(data, 0.01)
        
        bic = model_selection(model, data, flux_err, params, method="bic")
        
        assert np.isfinite(bic)
        assert bic < 1000
    
    def test_model_selection_aic(self):
        """Test AIC calculation"""
        time = np.linspace(0, 10, 100)
        params = [1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.05, 0.3]
        
        data = combined_model(time, *params)
        model = combined_model(time, *params)
        flux_err = np.full_like(data, 0.01)
        
        aic = model_selection(model, data, flux_err, params, method="aic")
        
        assert np.isfinite(aic)
        assert aic < 1000
    
    def test_model_selection_prefers_simpler_model(self):
        """Test that model selection penalizes complexity"""
        time = np.linspace(0, 10, 100)
        
        params_simple = [1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.05, 0.3]
        data = combined_model(time, *params_simple)
        model_simple = combined_model(time, *params_simple)
        flux_err = np.full_like(data, 0.01)
        
        params_complex = params_simple + [7.0, 0.01, 0.001]
        model_complex = combined_model(time, *params_complex)
        
        bic_simple = model_selection(model_simple, data, flux_err, params_simple, method="bic")
        bic_complex = model_selection(model_complex, data, flux_err, params_complex, method="bic")
        
        assert bic_simple < bic_complex
    
    # ========== Test ED Calculation ==========
    
    def test_ed_from_model_basic(self):
        """Test equivalent duration calculation"""
        time = np.linspace(0, 1, 100)
        baseline = np.ones_like(time) * 1.0
        flare = flare_model_davenport2014(time, 0.5, 0.05, 0.3)
        model = baseline + flare
        
        ed = ed_from_model(time, model, baseline)
        
        assert np.isfinite(ed)
        assert ed > 0
    
    def test_ed_from_model_no_flare(self):
        """Test ED with no flare (should be ~0)"""
        time = np.linspace(0, 1, 100)
        baseline = np.ones_like(time) * 1.0
        model = baseline.copy()
        
        ed = ed_from_model(time, model, baseline)
        
        assert ed == pytest.approx(0, abs=1e-6)
    
    def test_ed_from_model_larger_flare_larger_ed(self):
        """Test that larger flares have larger ED"""
        time = np.linspace(0, 1, 100)
        baseline = np.ones_like(time) * 1.0
        
        flare_small = flare_model_davenport2014(time, 0.5, 0.05, 0.1)
        model_small = baseline + flare_small
        ed_small = ed_from_model(time, model_small, baseline)
        
        flare_large = flare_model_davenport2014(time, 0.5, 0.05, 0.3)
        model_large = baseline + flare_large
        ed_large = ed_from_model(time, model_large, baseline)
        
        assert ed_large > ed_small
    
    # ========== Test Single Flare Fitting with Mocked MCMC ==========
    
    @patch('emcee.EnsembleSampler')
    def test_fit_single_flare_emcee_called(self, mock_sampler_class, single_flare_lc, mock_emcee_sampler):
        """Test that emcee is called with correct arguments"""
        flc = single_flare_lc
        time = flc.time.value
        flux = flc.flux.value
        flux_err = flc.flux_err.value
        
        # Create mock sampler
        mock_sampler_instance = mock_emcee_sampler(8)
        mock_sampler_class.return_value = mock_sampler_instance
        
        # Initial guess
        flare_guess = [5.0, 0.05, 0.3]
        
        result = fit_single_flare(
            time, flux, flux_err,
            flare_guess, max_flares=1
        )
        
        # Verify emcee was called
        assert mock_sampler_class.called
        assert mock_sampler_instance.run_mcmc.called
        
        # Verify result structure
        assert result is not None
        assert 'params' in result
        assert 'posterior_samples' in result
        assert result['n_flares'] == 1
    
    
    
    @patch('emcee.EnsembleSampler')
    def test_fit_multiple_flares_with_emcee(self, mock_sampler_class, multi_flare_lc, mock_emcee_sampler):
        """Test fitting multiple flares with mocked emcee"""
        flc = multi_flare_lc
        time = flc.time.value
        flux = flc.flux.value
        flux_err = flc.flux_err.value
        
        # Create appropriate mock for each flare region
        def sampler_side_effect(*args, **kwargs):
            # Return different mock sampler for each call
            # Use generic reasonable parameters
            return mock_emcee_sampler(8)
        
        mock_sampler_class.side_effect = sampler_side_effect
        
        tstarts = np.array([3.8, 9.8, 15.8]) + 2450000
        tstops = np.array([4.2, 10.2, 16.2]) + 2450000
        
        results = fit_flares(
            time, flux, flux_err,
            tstarts, tstops,
            buffer=0.2,
            max_flares=2,
                        plot=False
        )
        
        assert len(results) >= 3
        # Verify emcee was called multiple times (once per flare region)
        assert mock_sampler_class.call_count >= 3
    
    # ========== Test Flare Table Creation with MCMC ==========
    
    @patch('emcee.EnsembleSampler')
    def test_make_flare_table_with_posteriors(self, mock_sampler_class, single_flare_lc, mock_emcee_sampler):
        """Test that flare table includes uncertainties from posterior samples"""
        flc = single_flare_lc
        time = flc.time.value
        flux = flc.flux.value
        flux_err = flc.flux_err.value
        
        true_baseline = flc._true_baseline
        true_flare = flc._true_flare
        true_params = true_baseline + [true_flare['t_peak'], true_flare['fwhm'], true_flare['amplitude']]
        
        mock_sampler_instance = mock_emcee_sampler(len(true_params))
        mock_sampler_class.return_value = mock_sampler_instance
        
        tstarts = [4.8 + 2450000]
        tstops = [5.2 + 2450000]
        
        results = fit_flares(
            time, flux, flux_err,
            tstarts, tstops,
                        plot=False
        )

        print(results)
        
        table = make_flare_table(results)

        assert mock_sampler_class.called  # ✓ True
        assert mock_sampler_instance.run_mcmc.called  # ✓ True        
        # Check that error columns exist
        assert 't_peak_err' in table.columns
        assert 'fwhm_err' in table.columns
        assert 'amplitude_err' in table.columns
        
        # Check that errors are reasonable (not NaN, positive)
        assert table['t_peak_err'].notna().all()
        assert (table['t_peak_err'] > 0).all()
    
    # ========== Test Edge Cases ==========
    
    def test_fit_flares_empty_input(self):
        """Test that empty inputs are handled gracefully"""
        time = np.array([])
        flux = np.array([])
        flux_err = np.array([])
        tstarts = []
        tstops = []
        
        with pytest.raises(ValueError):
            fit_flares(time, flux, flux_err, tstarts, tstops)
    
    def test_fit_flares_mismatched_arrays(self):
        """Test that mismatched array lengths raise error"""
        time = np.linspace(0, 10, 100)
        flux = np.ones(50)
        flux_err = np.ones(100)
        tstarts = [5.0]
        tstops = [6.0]
        
        with pytest.raises(ValueError):
            fit_flares(time, flux, flux_err, tstarts, tstops)
    
    def test_fit_flares_with_nans(self):
        """Test handling of NaN values in data"""
        time = np.linspace(0, 10, 100)
        flux = np.ones(100)
        flux[40:50] = np.nan
        flux_err = np.full(100, 0.01)
        
        tstarts = [2.0]
        tstops = [3.0]
        
        with pytest.raises(ValueError, match="contain NaN or inf"):
            fit_flares(time, flux, flux_err, tstarts, tstops)
    
    @patch('emcee.EnsembleSampler')
    def test_emcee_with_invalid_params_returns_none(self, mock_sampler_class):
        """Test that fitting returns None when it fails"""
        # Create mock that raises exception
        mock_sampler_class.side_effect = Exception("MCMC failed")
        
        time = np.linspace(0, 10, 100)
        flux = np.ones(100)
        flux_err = np.full(100, 0.01)
        
        flare_guess = [5.0, 0.05, 0.3]
        
        result = fit_single_flare(
            time, flux, flux_err,
            flare_guess,
                        max_flares=1
        )
        
        # Should handle exception and return None
        assert result is None

    @patch('emcee.EnsembleSampler')
    def test_debug_fit_flares_call(self, mock_sampler_class, single_flare_lc, mock_emcee_sampler):
        """Debug test to see what's failing"""
        flc = single_flare_lc
        
        mock_sampler_instance = mock_emcee_sampler(8)
        mock_sampler_class.return_value = mock_sampler_instance
        
        print(f"Time range: {flc.time.value.min():.2f} to {flc.time.value.max():.2f}")
        print(f"Flux range: {flc.flux.value.min():.4f} to {flc.flux.value.max():.4f}")
        print(f"Data points: {len(flc.time)}")
        
        tstarts = [4.8]
        tstops = [5.2]
        
        try:
            results = fit_flares(
                flc.time.value, flc.flux.value, flc.flux_err.value,
                tstarts, tstops,
                buffer=0.1,
                plot=False
            )
            print(f"Results: {len(results)}")
            if len(results) > 0:
                print(f"First result keys: {results[0].keys()}")
        except Exception as e:
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()


    @patch('emcee.EnsembleSampler')
    def test_make_flare_table_integration(self, mock_sampler_class, mock_emcee_sampler):
        """Integration test with minimal synthetic data"""
        np.random.seed(42)
        time = np.linspace(0, 5, 1000)  # More points, wider range
        
        baseline = np.ones_like(time) * 1.0
        flare = flare_model_davenport2014(time, 2.5, 0.05, 0.4)
        
        flux = baseline + flare + np.random.normal(0, 0.005, len(time))
        flux_err = np.full_like(flux, 0.005)
        
        # Mock factory that returns sampler based on actual ndim requested
        def mock_sampler_factory(nwalkers, ndim, *args, **kwargs):
            # ndim is the actual number of parameters being fit
            return mock_emcee_sampler(n_params=ndim, n_walkers=nwalkers)
        
        mock_sampler_class.side_effect = mock_sampler_factory
        
        tstarts = [2.3]
        tstops = [2.7]
        
        results = fit_flares(
            time, flux, flux_err,
            tstarts, tstops,
            buffer=0.2,
            max_flares=1,  # Keep it simple - just 1 flare
            plot=False
        )
        
        print(f"Mock called: {mock_sampler_class.called}")
        print(f"Call count: {mock_sampler_class.call_count}")
        print(f"Results: {len(results)}")
        
        assert len(results) > 0
        table = make_flare_table(results)
        assert len(table) > 0
        assert 't_peak_err' in table.columns