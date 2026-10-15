import numpy as np
import pandas as pd
import os
import copy
import logging
import datetime
import warnings

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Column, Table
from astropy.time import Time
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from tqdm.auto import tqdm
        

from .flares import flare_factor
from .fit_flares import fit_flares as fit_flares_func
from .fit_flares import make_flare_table
from .fit_flares import plot_corner

from lightkurve import LightCurve
from lightkurve.utils import KeplerQualityFlags


from .altai import (find_flares,
                    _find_iterative_median)
from .fakeflares import (match_flares,
                         template_equivalent_duration,
                         generate_fake_flare_distribution,
                         mod_random,
                         flare_model,
                         )
from .injrecanalysis import wrap_characterization_of_flares, _heatmap
from .utils import split_cont_windows
from .utils import get_response_curve

LOG = logging.getLogger(__name__)


FLARE_COLUMNS = ['istart', 'istop', 'cstart', 'cstop', 'tstart',
                 'tstop', 'ed_rec', 'ed_rec_err', 'ampl_rec', 
                  'total_n_valid_data_points', 'dur']

FAKE_FLARE_COLUMNS = ['duration_d', 'amplitude', 'ed_inj', 'peak_time']


class FlareLightCurve(LightCurve):
    """
    Flare light curve class inheriting from Lightkurve's LightCurve class.

    Attributes inherited from LightCurve:
    -------------------------------------
    time : array-like
        Time measurements.
    flux : array-like
        Flux count for every time point.
    flux_err : array-like
        Uncertainty on each flux data point.
    meta : dict
        Metadata associated with the LightCurve. 

    Attributes specific to FlareLightCurve:
    ---------------------------------------
    meta["qcs"] : integer
        Quarter, Campaign, or Sector number.
    detrended_flux : array-like
        De-trended flux, relative to the quiescent level (see ``detrend``).
    detrended_flux_err : array-like
        Uncertainty of the de-trended flux.
    cont_windows : list of tuples of ints
        Continuous observing windows as (start, stop) index pairs, with stop
        exclusive. See ``find_cont_windows``.
    flares : DataFrame
        Table of flares, their start and stop time, recovered equivalent duration
        (ED), and, if applicable, recovery probability, ratio of recovered ED to
        injected synthetic ED. Also information about quality flags may be stored
        here.
    it_med : array-like
        Iterative median, see the ``find_iterative_median`` method.

    Methods that analyse the light curve change it in place: they fill the
    ``flares`` table or add columns (``detrended_flux``, ``gp_model``,
    ``it_med``) and attributes (``cont_windows``, ``fake_flares``), and
    return the light curve itself, so that calls can be chained.  Only
    ``inject_fake_flares``, whose result is a different light curve
    (synthetic flares in the flux), returns a new object and leaves this one
    unchanged.
    """


    @property
    def detrended_flux_err(self):
        """Uncertainty of the de-trended flux (``detrend``). None until computed."""
        return self["detrended_flux_err"] if "detrended_flux_err" in self.colnames else None

    @detrended_flux_err.setter
    def detrended_flux_err(self, detrended_flux_err):
        self["detrended_flux_err"] = detrended_flux_err


    @property
    def detrended_flux(self):
        """De-trended flux (``detrend``). None until computed."""
        return self["detrended_flux"] if "detrended_flux" in self.colnames else None

    @detrended_flux.setter
    def detrended_flux(self, detrended_flux):
        self["detrended_flux"] = detrended_flux 

    @property
    def gp_model(self):
        """Gaussian-process model of the variability (``detrend``). None until computed."""
        return self["gp_model"] if "gp_model" in self.colnames else None

    @gp_model.setter
    def gp_model(self, gp_model):
        self["gp_model"] = gp_model 


    @property
    def it_med(self):
        """Iterative median of the de-trended flux (``find_iterative_median``). None until computed."""
        return self["it_med"] if "it_med" in self.colnames else None

    @it_med.setter
    def it_med(self, it_med):
        self["it_med"] = it_med

    @property
    def origin(self):
        return self.meta.get("origin", "FLC")

    @origin.setter
    def origin(self, origin):
        self.meta["origin"] = origin 



    @property
    def flares(self) -> pd.DataFrame:
        """Flare table (``find_flares``); empty until flares are searched."""
        if "flares" in self.meta:
            return self.meta["flares"]
        return pd.DataFrame(columns=FLARE_COLUMNS)

    @flares.setter
    def flares(self, flares):
        self.meta["flares"] = flares 


    @property
    def fake_flares(self) -> pd.DataFrame:
        """Injection-recovery table (``sample_flare_recovery``); empty until
        filled."""
        if "fake_flares" in self.meta:
            return self.meta["fake_flares"]
        return pd.DataFrame(columns=FAKE_FLARE_COLUMNS)

    @fake_flares.setter
    def fake_flares(self, fake_flares):
        self.meta["fake_flares"] = fake_flares


    @property
    def cont_windows(self):
        """Continuous observing windows (``find_cont_windows``); None until
        computed."""
        return self.meta.get("cont_windows")

    @cont_windows.setter
    def cont_windows(self, cont_windows):
        self.meta["cont_windows"] = cont_windows

    @property
    def gaps(self):
        """Deprecated alias of ``cont_windows``."""
        warnings.warn("FlareLightCurve.gaps is deprecated; use cont_windows.",
                      DeprecationWarning, stacklevel=2)
        return self.cont_windows

    @gaps.setter
    def gaps(self, cont_windows):
        warnings.warn("FlareLightCurve.gaps is deprecated; use cont_windows.",
                      DeprecationWarning, stacklevel=2)
        self.cont_windows = cont_windows

    @property
    def cadenceno(self):
        """Cadence number for each observation; 0, 1, 2, ... (not stored)
        if the light curve has no cadence numbers."""
        if "cadenceno" in self.colnames:
            return self["cadenceno"]
        return Column(np.arange(len(self.time), dtype=int), name="cadenceno")

    @cadenceno.setter
    def cadenceno(self, cadenceno):
        self["cadenceno"] = cadenceno

    def _init_flare_table(self, flares=None, fake_flares=None):

        if flares is None:
            self.flares = pd.DataFrame(columns=FLARE_COLUMNS)
        else:
            self.flares = flares

        if fake_flares is None:
            
            self.fake_flares = pd.DataFrame(columns=FAKE_FLARE_COLUMNS)
        else:
            self.fake_flares = fake_flares


    def __repr__(self):
        mission = self.meta.get("mission", "Unknown")
        qcs = self.meta.get("qcs", "Unknown")

        try:
            targetid = self.targetid
        except (AttributeError, KeyError):
            targetid = "Unknown"
        
        return(f'FlareLightCurve(ID: {targetid!s:<9} |' \
            f' Mission: {mission!s:<6} |' \
            f' QCS: {qcs!s:>3} |' \
            f' Cadence: {self._cadence_str()})')

    def _qcs(self):
        """Quarter, campaign or sector number from the metadata, in any
        capitalisation, or "Unknown"."""
        for key in ("qcs", "campaign", "sector", "quarter"):
            for k in (key, key.upper()):
                if self.meta.get(k) is not None:
                    return self.meta[k]
        return "Unknown"

    def _cadence_str(self):
        """Cadence in seconds as a string: from ``meta["cadence"]`` if it is
        a number, else the median time step, else "Unknown"."""
        cadence = self.meta.get("cadence")
        if not isinstance(cadence, (int, float, np.integer, np.floating)) or not np.isfinite(cadence):
            if len(self.time) > 1:
                cadence = float(np.nanmedian(np.diff(self.time.value))) * 86400.
            else:
                return "Unknown"
        return f"{cadence:.0f} s"

    
    def __str__(self):
        return self.__repr__()  
    
    def _repr_html_(self):
        return f"<pre>{self.__repr__()}</pre>"

    def save_to_fits(self, loc=None, name=None, overwrite=True):
        """
        Save FlareLightCurve to a FITS file.
        
        Parameters
        ----------
        loc : str
            Path to folder to save the FITS file. If None, current directory.
        name : str, optional
            Name of the FITS file. If None, default name.
        """
        if loc is None:
            loc = os.getcwd()
        if name is None:
            name = f"flc_{self.targetid}_" \
                f"{self.meta.get('mission','Unknown')}_" \
                f"{self.meta.get('qcs','Unknown')}_" \
                f"{self._cadence_str().replace(' s', '')}.fits"
        
        path = os.path.join(loc, name)
        
        # Add targetid and qcs to metadata
        self.meta['TARGETID'] = self.targetid
        self.meta['QCS'] = self.meta.get('qcs', 'Unknown')

        extra_columns = {}
        for col in ['detrended_flux', 'detrended_flux_err', 'it_med']:
            self._add_column_if_valid(extra_columns, col)
        
        # Create FITS HDU
        hdul = self.to_fits(flux_column_name='flux', **extra_columns)
        
        # Add simple metadata entries to PRIMARY header (HDU 0)
        for key, value in self.meta.items():
            # Only add simple types that FITS can handle
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                try:
                    # FITS header keys must be 8 chars or less, convert to uppercase
                    fits_key = key[:8].upper()
                    hdul[0].header[fits_key] = value
                except:
                    pass
        
        # Write to file
        hdul.writeto(path, overwrite=overwrite)
        
        return path

    def _add_column_if_valid(self, extra_columns, col_name):
        """Add column to dict if it exists and contains non-NaN data."""
        try:
            if col_name in self.colnames and not np.all(np.isnan(self[col_name])):
                extra_columns[col_name] = self[col_name]
        except (KeyError, AttributeError):
            pass

    @classmethod
    def read_from_fits(cls, path):
        """
        Read a FlareLightCurve from a FITS file.
        
        Parameters
        ----------
        path : str
            Path to the FITS file.
        
        Returns
        -------
        flc : FlareLightCurve
            FlareLightCurve object loaded from the FITS file.
        """
        
        # Suppress UnitsWarning for non-standard FITS units
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=AstropyWarning, 
                                    message='.*did not parse as fits unit.*')
            
            # Read the FITS file as a table
            table = Table.read(path, format='fits', hdu=1)
        
        # Normalize all column names to lowercase
        for col in table.colnames:
            table.rename_column(col, col.lower())
        
        # Extract metadata from PRIMARY header (HDU 0)
        with fits.open(path) as hdul:
            header = hdul[0].header
            meta = {}
            for key in header.keys():
                if key:
                    meta[key.lower()] = header[key]
        
        # Find and extract required columns (now all lowercase)
        time_col = flux_col = flux_err_col = None
        
        for col in table.colnames:
            if col == 'time':
                time_col = col
            elif col == 'flux':
                flux_col = col
            elif col == 'flux_err':
                flux_err_col = col
        
        if time_col is None:
            raise ValueError(f"No time column found. Available: {table.colnames}")
        if flux_col is None:
            raise ValueError(f"No flux column found. Available: {table.colnames}")
        
        # Extract time, flux, flux_err
        time_array = table[time_col]
        flux_data = table[flux_col]
        flux_err_data = table[flux_err_col] if flux_err_col else None
        
        # Convert time to a Time object.  save_to_fits stores the time format
        # as the column's unit: "btjd" (TESS) or "bkjd" (Kepler/K2), which
        # lightkurve registers as astropy Time formats.  Anything else is
        # read as JD.
        time_unit_str = str(getattr(time_array, 'unit', '')).lower()
        time_values = np.asarray(getattr(time_array, 'value', time_array))
        if time_unit_str in ('btjd', 'bkjd'):
            time_data = Time(time_values, format=time_unit_str)
        else:
            time_data = Time(time_values, format='jd')
        
        # Handle non-standard flux units
        if hasattr(flux_data, 'unit') and str(flux_data.unit) == 'electron / s':
            flux_data = flux_data.value * (u.electron / u.s)
        
        if flux_err_data is not None and hasattr(flux_err_data, 'unit') and str(flux_err_data.unit) == 'electron / s':
            flux_err_data = flux_err_data.value * (u.electron / u.s)
        
        # Remove these columns from table (will pass separately)
        table.remove_column(time_col)
        table.remove_column(flux_col)
        if flux_err_col:
            table.remove_column(flux_err_col)
        
        # Create FlareLightCurve with explicit time, flux, flux_err
        flc = cls(time=time_data, flux=flux_data, flux_err=flux_err_data, 
                data=table, meta=meta)
        
        # Restore targetid
        if 'targetid' in meta:
            flc.targetid = meta['targetid']
        else:
            warnings.warn("TARGETID not found in FITS header.", UserWarning)
        
        return flc
    
    def find_iterative_median(self, n=10, detrended=True, per_window=False, **kwargs):
        """
        Find the iterative median of the (detrended) flux, using sigma
        clipping to exclude outliers such as flares.

        By default a single median is computed for the whole light curve:
        the baseline rarely shifts between continuous observing windows,
        while stellar variability can bias the median of a short window.

        Parameters
        ----------
        n : int, optional
            Maximum number of sigma-clipping iterations. Default is 10.
        detrended : bool, optional
            Use ``detrended_flux`` (default) or ``flux``.
        per_window : bool, optional
            If True, compute one median per continuous window (see
            ``find_cont_windows``; the windows are found first if needed),
            e.g. when the baseline jumps between windows. Default False.
        **kwargs : dict
            Keyword arguments to pass to sigma_clip

        Returns
        -------
        self : FlareLightCurve
            Returns self with it_med attribute updated
            
        Examples
        --------
        >>> flc.find_iterative_median()
        >>> # One median per continuous window:
        >>> flc.find_iterative_median(per_window=True)
        """
        # Extract arrays from self
        if detrended == True:
            detrended_flux = self.detrended_flux
            if detrended_flux is None:
                raise ValueError("No detrended flux: call detrend() first, or use "
                                 "find_iterative_median(detrended=False).")
        else:
            detrended_flux = self.flux

        # None makes _find_iterative_median use the whole light curve
        cont_windows = None
        if per_window:
            if self.cont_windows is None:
                self.cont_windows = self.find_cont_windows().cont_windows
            cont_windows = self.cont_windows

        # Call internal function
        it_med = _find_iterative_median(detrended_flux, cont_windows, n, **kwargs)
        
        # Set result on self
        self["it_med"] = it_med

        
        return self
    
    def find_cont_windows(self, maxgap=0.09, minspan=10, splits=[]):
        '''
        Find the continuous observing windows of the light curve, i.e. the
        stretches between time gaps of at least ``maxgap``, and store them in
        the ``cont_windows`` attribute as (start, stop) index pairs with stop
        exclusive. Windows shorter than ``minspan`` cadences are dropped.
        If required, adds additional splits in an arbitrary number of places.
        Caution: passing splits values means that you override the minspan
        and maxgap.

        Parameters
        ------------
        time : numpy array with floats
            sorted array, in units of days
        maxgap : 0.09 or float
            maximum time gap between two datapoints in days,
            default equals approximately 2h
        minspan : 10 or int
            minimum number of datapoints in continuous observation,
            i.e., w/o gaps as defined by maxgap
        splits : list of floats or ints
            additional places in which to slice the time series

        Returns
        --------
        FlareLightCurve
            The light curve itself, with ``cont_windows`` set.
        '''
        lc = self
        dt = np.diff(lc.time.value)
        gap = np.where(np.append(0, dt) >= maxgap)[0]

        # add start/end of LC to loop over easily
        gap_out = np.append(0, np.append(gap, len(lc.time.value)))

        # left start, right end of data
        left, right = gap_out[:-1], gap_out[1:]

        # drop too short observation periods
        too_short = np.where(np.diff(gap_out) < minspan)
        left, right = np.delete(left,too_short), np.delete(right,(too_short))

        # get the continuous windows
        cont_windows = list(zip(left, right))
        
        # split up the time series in additional place if needed
        lc.meta["cont_windows"] = split_cont_windows(cont_windows, splits)

        return lc

    def find_gaps(self, *args, **kwargs):
        """Deprecated alias of ``find_cont_windows``."""
        warnings.warn("FlareLightCurve.find_gaps is deprecated; use find_cont_windows.",
                      DeprecationWarning, stacklevel=2)
        return self.find_cont_windows(*args, **kwargs)

    def detrend(self, mode=None, func=None, save=False,
                path='detrended_lc.fits', **kwargs):
        """
        De-trend the light curve, by default with
        ``altaipony.detrending.custom_detrending``.

        Examples
        --------
        >>> flc.detrend()                  # custom_detrending, default settings
        >>> flc.detrend(gp_bin_factor=5)   # options of custom_detrending
        >>> flc.detrend(func=my_function)  # your own de-trending function

        Parameters
        ----------
        mode : None
            Deprecated. ``"custom"`` is still accepted (with a warning) for
            calls of the form ``detrend("custom", func=...)``.
        func : function or None
            De-trending function: takes a FlareLightCurve (and ``kwargs``) and
            returns it with ``detrended_flux`` and ``detrended_flux_err`` set,
            on the same time stamps. Default ``custom_detrending``.
        save : False or bool
            If True, the light curve is saved as a fits file to ``path``.
        path : str
            Path to resulting fits file.
        kwargs : dict
            Keyword arguments to pass to ``func``.

        Returns
        --------
        FlareLightCurve
            The light curve itself, with the columns the detrending function
            produced (e.g. ``detrended_flux``, ``detrended_flux_err``,
            ``gp_model``, ``it_med``) added or updated.
        """
        if mode is not None:
            if mode != "custom":
                raise ValueError(f'De-trending mode {mode!r} does not exist. Call '
                                 'detrend() for custom_detrending, or pass your own '
                                 'function as func=.')
            warnings.warn('detrend("custom", func=...) is deprecated; call detrend() '
                          'or detrend(func=...).', DeprecationWarning, stacklevel=2)
        if func is None:
            from .detrending import custom_detrending
            func = custom_detrending

        # The function works on a copy: it may restructure its input (e.g.
        # interpolate missing cadences); its results are then copied back.
        new_lc = func(copy.deepcopy(self), **kwargs)

        if (new_lc.detrended_flux is None or new_lc.detrended_flux_err is None
                or np.isnan(new_lc.detrended_flux).all()
                or np.isnan(new_lc.detrended_flux_err).all()):
            raise AttributeError('The de-trending function does not return a '
                                 'detrended_flux or detrended_flux_err attribute.')

        if len(new_lc) != len(self) or not np.allclose(new_lc.time.value, self.time.value):
            raise ValueError('The de-trending function returned a light curve with '
                             'different time stamps; its results cannot be added '
                             'to this light curve.')
        for col in new_lc.colnames:
            if col != "time":
                self[col] = new_lc[col]
        for key, value in new_lc.meta.items():
            if key not in ("flares", "fake_flares"):
                self.meta[key] = value

        if save == True:
            self.to_fits(path)

        return self


            
            
    def get_energies(self, teff, radius, resp=None, wav=None, mission=None, response_path=None, path=None):
        
        """
        Compute and add bolometric flare energies to the flare table.

        Parameters
        ----------
        teff : float
            Effective temperature of the star [K]
        radius : float
            Stellar radius [R_sun]
        resp : array-like, optional
            Instrument response curve (same length as wav)
        wav : array-like, optional
            Wavelengths corresponding to resp [Angstrom]
        mission : str, optional
            Instrument name: 'kepler' or 'tess'
        response_path : str, optional
            Path to CSV file with 'lambda' and 'resp' columns
        path : str, optional
            Alias for response_path

        Raises
        ------
        ValueError
            If no flares are found, input arrays mismatch, or mission is invalid.

        Returns
        -------
        FlareLightCurve
            The light curve object with 'bolometric_energy_erg' column added.
        """

        if self.flares.empty:
            raise ValueError("No flares found. Please run find_flares() before computing energies.")


        # Support 'path=' as alias for 'response_path'
        if path and not response_path:
            response_path = path

        # Check if both direct arrays and indirect loading are passed
        if (wav is not None or resp is not None) and (mission or response_path):
            warnings.warn("Ignoring mission/path because 'wav' and 'resp' were provided directly.")

        # Validate custom response arrays
        if wav is not None and resp is not None:
            if len(wav) != len(resp):
                raise ValueError("wav and resp must have the same length.")
            if len(wav) < 10:
                raise ValueError("wav and resp must contain at least ~10 points.")
        else:
            # Load from mission or file path
            mission = mission or self.meta.get("mission", "")
            wav, resp = get_response_curve(mission=mission, custom_path=response_path)

        # Compute and apply flare factor
        ff = flare_factor(teff, radius, wav, resp).value
        self.flares["bolometric_energy_erg"] = self.flares["ed_rec"] * ff

        return self


        
            
            
            
            
            
    def find_flares(self, minsep=3, fake=False, maxgap=0.09, median_iterations=4,
                    **kwargs):

        '''
        Find flares in a ``FlareLightCurve``.

        Parameters
        -------------
        minsep : 3 or int
            Minimum distance between two candidate start times in datapoints.
        fake : bool
            Kept for compatibility; flares are always searched anew, and any
            previous flare table is replaced.
        maxgap : float (Default is 0.09)
            Time gap in days that separates continuous observing windows
            (see ``find_cont_windows``).
        median_iterations : int (Default is 4)
            Maximum number of sigma-clipping iterations of the iterative
            median (see ``find_iterative_median``).
        kwargs : dict
            keyword arguments to pass to :func:`find_flares_in_cont_obs_period`
        
        Possible keyword arguments: 
   	    
        sigma : numpy array
            local scatter of the flux. Array should be the same length as the
            detrended flux array. 
            If sigma=None, error is used instead.
        N1 : int (default is 3)
            How many times above sigma is required.
        N2 : int (Default is 2)
            How many times above sigma and detrended_flux_err is required
        N3 : int (Default is 3)
            The number of consecutive points required to flag as a flare.
        max_window : int or None (Default is 10)
            Longest window, in cadences, of the window detector: windows of
            ``N3`` to ``max_window`` points whose combined probability of
            being noise is below that of ``N3`` points at ``N1`` sigma are
            flagged. None gives the classic rule (``N3`` consecutive points
            above the thresholds, Chang et al. 2015).
        beta_window_days : float (Default is 1.0)
            Time span over which the window detector corrects for correlated
            residuals.
        merge_sigma : float or None (Default is 0.25)
            Merge consecutive candidates when the mean flux between them
            stays at least ``merge_sigma`` times the noise above the
            iterative median, so that long, faint flares are not split into
            several candidates. None disables merging.
        extend_sigma : float or None (Default is 1.0)
            Extend each detection over adjacent cadences at least
            ``extend_sigma`` times the noise above the iterative median.
            None disables the extension.

        Returns
        ----------
        FlareLightCurve
            The light curve itself, with the ``flares`` table replaced and
            ``cont_windows`` and ``it_med`` updated.
        '''
        lc = self
        # replace any previous flare table
        lc.flares = pd.DataFrame(columns=FLARE_COLUMNS)
        # find continuous observing periods
        lc = lc.find_cont_windows(maxgap=maxgap)
        # find the true median value iteratively
        lc = lc.find_iterative_median(n=median_iterations)
        # find flares (the module function works on a copy)
        lc.flares = find_flares(lc, minsep=minsep, **kwargs).flares

        return lc
    

    def highlight_all_flares_in_lightcurve(self, save=None):
        """
        Plots the light curve with all detected flares highlighted.
        
        Parameters
        ----------
        save : None or str
            If a string is provided, the plot is saved to the given path.

        Returns
        -------
        None"""
        # Plot light curve with flare regions highlighted
        plt.figure(figsize=(12, 5))
        plt.plot(self.time.value, self.flux, alpha=0.7)

        # Highlight flare windows with translucent red spans
        for t1, t2 in self.flares[['tstart', 'tstop']].values:
            plt.axvspan(t1, t2, color='red', alpha=0.2)

        plt.xlabel("Time [BTJD]")
        plt.ylabel("Flux [e⁻/s]")
        plt.xlim(self.time.value.min(), self.time.value.max())
        plt.title("Flares in light curve")
        plt.tight_layout()
        plt.savefig(save) if save else plt.show()

    def zoom_in_on_flare(self, ind, buffer=0.075, save=None):
        """
        Plots a zoomed-in view of a specific flare in the light curve.

        Parameters
        ----------
        ind : int
            Index of the flare to zoom in on.
        save : None or str
            If a string is provided, the plot is saved to the given path.

        Returns
        -------
        None
        """
    
        t1,t2 = self.flares[['tstart', 'tstop']].iloc[ind]

        # Define zoom window with buffer
        t_min = t1 - buffer
        t_max = t2 + buffer

        # Select data in zoomed window
        mask = (self.time.value >= t_min) & (self.time.value <= t_max)
        time_zoom = self.time.value[mask]
        flux_zoom = self.flux[mask]  

        # Plot zoomed region
        plt.figure(figsize=(10, 4))
        plt.plot(time_zoom, flux_zoom, alpha=0.8)
        plt.axvspan(t1, t2, color='red', alpha=0.3, 
                          label='Flare region')
        plt.title(f'Zoom on flare {ind}')
        plt.xlim(time_zoom[0], time_zoom[-1])
        plt.xlabel("Time [BTJD or BKJD]")
        plt.ylabel("Flux [e-/s]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save) if save else plt.show()


    def sample_flare_recovery(self, iterations=2000, inject_before_detrending=False,
                              mode=None, func=None, save_lc_to_file=False, folder="", 
                              fakefreq=0.05, save=False, path=None, detrend_kwargs=None,
                              progress=True, **kwargs):
        """
        Runs a number of injection recovery cycles and characterizes the light
        curve by recovery probability and equivalent duration underestimation.
        Inject one flare per light curve.

        Parameters
        -----------
        iterations : 2000 or int
            Number of injection/recovery cycles
        inject_before_detrending : False or bool
            If True, fake flare are injected directly into raw data.
        mode : None
            Deprecated, ignored (de-trending uses ``func``).
        func : function or None
            De-trending function passed to ``FlareLightCurve.detrend`` when
            ``inject_before_detrending`` is True. Default
            ``custom_detrending``.
        fakefreq : 0.05 or float
            number of flares per day, but at least one per continuous observation period will be injected
        detrend_kwargs : dict
            Keyword arguments to pass to FlareLightCurve.detrend
        progress : True or bool
            Show a progress bar with the number of fake flares injected and
            recovered so far.
        kwargs : dict
            Keyword arguments to pass to inject_fake_flares

        Returns
        -------
        lc : FlareLightCurve
            The light curve itself (detrended, if ``inject_before_detrending``),
            with its flares found and all injection-recovery results in
            ``fake_flares``.
        fake_lc : FlareLightCurve
            Light curve with the last iteration of synthetic flares injected.
        """
        injrecstr = {True : "before", False : "after"} # define string to identify fake flare analysis by file name
        detrend_kwargs = {} if detrend_kwargs is None else detrend_kwargs
        if mode is not None:
            warnings.warn("sample_flare_recovery(mode=...) is deprecated and ignored; "
                          "pass func= to choose the de-trending function.",
                          DeprecationWarning, stacklevel=2)
        
        lc = self
        if inject_before_detrending == True:
            lc.detrend(func=func, **detrend_kwargs)
        lc.find_cont_windows()
        lc.find_flares()
        lc.find_iterative_median()
        
        columns =  ['istart', 'istop', 'cstart', 'cstop', 'tstart', 'tstop',
                    'ed_rec', 'ed_rec_err', 'duration_d', 'amplitude', 'ed_inj',
                    'peak_time', 'ampl_rec', 'dur', 'recovered']
        

        n_injected, n_recovered = 0, 0
        bar = tqdm(range(iterations), desc="Injection-recovery", unit="iteration",
                   disable=not progress)
        for i in bar:
            fake_lc = lc.inject_fake_flares(inject_before_detrending=inject_before_detrending,
                                                 fakefreq=fakefreq,
                                                 **kwargs)
            
            if save_lc_to_file == True:
                fake_lc.to_fits(f"{folder}before.fits")
                print(f"saved {self.targetid} LC before detrending")
                
            injs = fake_lc.fake_flares
           
            if inject_before_detrending == True:
                LOG.info('\nDetrending fake LC:\n')
                fake_lc = fake_lc.detrend(func=func, **detrend_kwargs)
            
            fake_lc = fake_lc.find_flares(fake=True)
            recs = fake_lc.flares
          
            if save_lc_to_file == True:
                fake_lc.to_fits(f"{folder}after.fits")
                print(f"saved {self.targetid} LC after detrending")
                
            # The following line makes sure that column order is retained    
            injrec_results = pd.DataFrame(columns=columns)
            
            # Merge injected and recovered flares
            injrec_results = pd.concat([injrec_results, match_flares(injs, recs)[0]],
                                                   ignore_index=True)
            

            n_injected += len(injrec_results)
            n_recovered += int(injrec_results["recovered"].fillna(False).astype(bool).sum())
            bar.set_postfix(injected=n_injected, recovered=n_recovered)
            
            if save == True:
            
                #Define default path if needed
                if path is None:
                    path = (f'{iterations}_{lc.targetid}_inj_'
                            f'{injrecstr[inject_before_detrending]}_'
                            f'{lc._qcs()}.csv')
                            
                # If it already exists append new injrec to the end       
                if os.path.exists(path):
                    with open(path, 'a') as f:
                        injrec_results.to_csv(f, index=False, header=False)
                # If it doesn't then write it out but keep the header
                else:
                    injrec_results.to_csv(path, index=False)
            
            # Add to previous runs of sample_flare_recovery on the same LC or create new table    
            if lc.fake_flares.shape[0] > 0:    
                lc.fake_flares = pd.concat([lc.fake_flares,injrec_results], ignore_index=True)
            else:
                lc.fake_flares = injrec_results
                
    
        if save == True:
            # Finally read in the result                    
            lc.fake_flares = pd.read_csv(path)  

        
        bar.close()
        return lc, fake_lc

 
    def mark_flagged_flares(self, explain=False):
        """
        Mark all flares that coincide with flagged cadences.
        Explain the flags if needed.

        Parameters
        -----------
        explain : False or bool
            If True, an ``explanation`` column will be added to the flares table
            explaining the flags that were raised during the flare duration.

        Returns
        --------
        FlareLightCurve (itself) with the flares table supplemented with an
        integer ``quality`` and, if applicable, a string ``explanation``
        column.
        """
        lc = self
        f = lc.flares
        if 'quality' not in f.columns:
            f['quality'] = 0
        f.quality = f.apply(lambda x: np.sum(lc.quality[x.istart:x.istop],
                                             dtype=int),
                            axis=1)
        if explain == True:
            g = lambda x: ', '.join(KeplerQualityFlags.decode(x.quality))
            f['explanation'] = f.apply(g, axis=1)
        lc.flares = f
        return lc

    def inject_fake_flares(self, model="mendoza2022", gapwindow=0.1, fakefreq=.005,
                           inject_before_detrending=False, d=False, seed=None, 
                           **kwargs):
        '''
        Create a number of events, inject them in to data
        Use grid of amplitudes and durations, keep ampl in relative flux units
        Keep track of energy in Equiv Dur.
        Duration defined in minutes
        Amplitude defined multiples of the median error


        Parameters:
        -------------
        model : "mendoza2022" or "davenport2014"
            The flare model to use. Default is "mendoza2022".
        gapwindow : 0.1 or float

        fakefreq : .005 or float
            flares per day, but at least one per continuous observation period will be injected
        inject_before_detrending : True or bool
            By default, flares are injected before the light curve is detrended.
        d : boolean
            If True, a seed for random numbers will be set
        seed : int or None
            If d == True, seed will be set to this number 
        kwargs : dict
            Keyword arguments to pass to generate_fake_flare_distribution.

        Returns:
        ------------
        FlareLightCurve with fake flare signatures

        '''

        fake_lc = copy.deepcopy(self)
        LOG.debug(str() + '{} FakeFlares started'.format(datetime.datetime.now()))
        
        # Either inject flares into the un-detrended light curve
        if inject_before_detrending == True:
            typ, typerr = 'flux', 'flux_err'
            LOG.debug('Injecting before detrending.')
            
        # ... or into the detrended one
        elif inject_before_detrending == False:
            typ, typerr = 'detrended_flux', 'detrended_flux_err'
            LOG.debug('Injecting after detrending.')
        
        # How many flares do you want to inject
        # At least one per continuous window
        # or as defined by the frequency
        nfakesum = max(len(fake_lc.cont_windows),
                       int(np.rint(fakefreq *
                           (fake_lc.time.value.max() - fake_lc.time.value.min()))
                           )
                       )
        
        # Use a light curve where you know the median flux: of the flux the
        # flares go into (raw flux if injected before detrending)
        fake_lc = fake_lc.find_iterative_median(detrended=(typ == "detrended_flux"))
        
        # Init arrays for the synthetic flare parameters
        t0_fake = np.zeros(nfakesum, dtype='float') # peak times
        ed_fake = np.zeros(nfakesum, dtype='float') # ED
        dur_fake = np.zeros(nfakesum, dtype='float') # duration
        ampl_fake = np.zeros(nfakesum, dtype='float') # amplitude
        
        # Init the synthetic flare counter to allow to point to the right
        # places in the arrays above (XXX_fake etc.)
        ckm = 0
        
        # Iterate over continuous observing periods
        for (le,ri) in fake_lc.cont_windows:
            
            # Pick the observing period
            gap_fake_lc = fake_lc[le:ri]
            
            # Define the number of synthetic flares you want to inject
            # minimum of 1
            nfake = max(1, int(np.rint(fakefreq *
                                       (gap_fake_lc.time.value.max() -
                                        gap_fake_lc.time.value.min()
                                       )
                                      )
                              )
                        )

            LOG.debug(f'Inject {nfake} fake flares into a {ri-le} datapoint long array.')
            
            # Are there real flares to deal with in the gap?
            real_flares_in_gap = self.flares[(self.flares.istart >= le) &
                                             (self.flares.istop <= ri)]
                                             
            # Pick flux, time, and flux error arrays 
            error = gap_fake_lc[typerr]
            flux = gap_fake_lc[typ]

            # account for different data types for detrended and non-detrended data
            if typ == "flux":
                flux = flux.value
                error = error.value

            time = gap_fake_lc.time.value
            
            # generate the time constraints for the flares you want to inject
            mintime, maxtime = np.min(time), np.max(time)
            dtime = maxtime - mintime
            
            # generate a distribution of durations and amplitudes
            distribution  = generate_fake_flare_distribution(nfake, d=d,
                                                            seed=seed, **kwargs)
            # add the distribution for this observing period 
            # to the full list of injected flares
            dur_fake[ckm:ckm+nfake], ampl_fake[ckm:ckm+nfake] = distribution
            
            # loop over the numer of fake flares you want to generate
            for k in range(ckm, ckm+nfake):
                
                # generate random peak time, avoid known flares
                isok = False
                
                # keep picking new random peak times for your synthetic flares
                # until it does not overlap with a real one
                while isok is False:
                
                    # choose a random peak time
                    # if you set a seed you will get the same synthetic flares
                    # all the time
                    if isinstance(seed, int):
                        t0 = (mod_random(1, d=d, seed=seed * k) * dtime + mintime)[0]
                        
                    # if you do note set a seed, the synthetic flares will be
                    # randomly distributed
                    elif seed is None:
                        t0 = (mod_random(1, d=d) * dtime + mintime)[0]
                     
                    # Check if there are there any real flares to deal with
                    # at that peak time. Only relevant if there were any flares
                    # detected at all:
                    if real_flares_in_gap.tstart.shape[0] > 0:
                    
                        # Are there any real flares happening at peak time?
                        # Fake flares should not overlap with real ones.
                        b = (real_flares_in_gap[(t0 >= real_flares_in_gap.tstart) &
                                                (t0 <= real_flares_in_gap.tstop)].
                                                 shape[0] )
                                                 
                        # number of flares that overlap should be 0
                        if b == 0:
                            isok = True
                            
                    # No real flares, no trouble:
                    else:
                        isok = True
                        
                    # add the peak time to the list    
                    t0_fake[k] = t0
                    
                    # generate the flare flux from the Davenport 2014 model
                    fl_flux = flare_model(model,time, t0, dur_fake[k], ampl_fake[k])
                    
                    # calculate the injected ED
                    ed_fake[k] = template_equivalent_duration(time, fl_flux)
                    
                # inject flare in to light curve by adding the flare flux
                if typ == "flux":
                    fake_lc[typ].value[le:ri] = (fake_lc[typ][le:ri].value +
                                                fl_flux * fake_lc.it_med[le:ri])
                elif typ == "detrended_flux":
                    fake_lc[typ][le:ri] = (fake_lc[typ][le:ri] +
                                                fl_flux * fake_lc.it_med[le:ri])
            # Increment the counter
            ckm += nfake
            
        # error minimum is a safety net for the spline function if mode=3
        fake_lc[typerr] = max( 1e-10, np.nanmedian( pd.Series(fake_lc[typ]).
                                                rolling(3, center=True).
                                                std() ) )*np.ones_like(fake_lc[typ])
        # Put the data together
        injected_events = {'duration_d' : dur_fake,
                           'amplitude' : ampl_fake,  
                           'ed_inj' : ed_fake,
                           'peak_time' : t0_fake}

        # the fake_flares attribute is a pandas DataFrame
        fake_lc.fake_flares = pd.DataFrame(injected_events)
        
        # Free up space
        del dur_fake
        del ampl_fake
        
        # Return the FLC with the injected flares
        return fake_lc

    def load_injrec_data(self, path, **kwargs):
        """Fetch the injection-recovery table
        from a given path, and append it to 
        any existing table.

        Parameters:
        -----------
        path : string
            path to file
        kwargs : dict
            keyword arguments to pass to
            `pandas.read_csv()`
        """

        df = pd.read_csv(path)
        if self.fake_flares.shape[0]>0:
            LOG.warning("The file is appended to an existing table.")
            self.fake_flares = pd.concat([self.fake_flares, df])
        else:
            self.fake_flares = df
        return self

    def plot_recovery_probability_heatmap(self, ampl_bins=None, 
                                          dur_bins=None, flares_per_bin=20, **kwargs):
        """Plot injected amplitude and injected
        FWHM vs. the fraction of recovered flares.
        
        Parameters:
        -----------
        ampl_bins : int or array
            bins for amplitudes
        dur_bins : int or array
            bins for FWHM
        flares_per_bin : int
            number of flares per bin, default is 20
        """

        flc = copy.deepcopy(self)
        return _heatmap(flc, "recovery_probability", 
                        ampl_bins, dur_bins, flares_per_bin, **kwargs)

    def plot_ed_ratio_heatmap(self, ampl_bins=None, dur_bins=None, flares_per_bin=20, **kwargs):
        """Plot recovered amplitude and recovered
        duration vs. the ratio of recovered ED to
        injected ED.
        
        Parameters:
        -----------
        ampl_bins : int or array
            bins for recovered amplitudes
        dur_bins : int or array
            bins for recovered duration
        flares_per_bin : int
            number of flares per bin, default is 20
        """
        flc = copy.deepcopy(self)
        return _heatmap(flc, "ed_ratio", 
                        ampl_bins, dur_bins, flares_per_bin, **kwargs)

    def characterize_flares(self, flares_per_bin=30, ampl_bins=None, dur_bins=None):
        """Use results from injection recovery to determine
        corrected flare characteristics, added as columns to the
        ``flares`` table (in place; returns the light curve itself).
        """
        flc = self
        flares = wrap_characterization_of_flares(flc.fake_flares.copy(), flc.flares.copy(),
                                                 flares_per_bin=flares_per_bin,
                                                 ampl_bins=ampl_bins,
                                                 dur_bins=dur_bins)
        flc.flares = flares
        return flc
    


            
    def fit_flares(self, buffer=0.05, max_flares=3, delta_bic=0.0, plot=True, debug_plot=False, **kwargs):
        """
        Fit flares using a polynomial baseline + analytic flare model (Davenport et al. 2014).

        This method automatically uses:
        - time
        - flux and flux_err
        - 'tstart' and 'tstop' as fitting regions

        Parameters
        ----------
        buffer : float
            Time buffer in days added before/after each flare region (default: 0.05).
        max_flares : int
            Maximum number of flare components to try per region (default: 3).
        plot : bool
            Whether to show plots for each best-fit flare region (default: True).
        debug_plot : bool
            Show all intermediate trial models (useful for debugging; default: False).
        **kwargs : dict
            Additional keyword arguments passed to the fitting backend.

        Returns
        -------
        FlareLightCurve
            The light curve itself; the fitted model results are stored and
            summarised by ``flare_table()``.
        """
        results = fit_flares_func(
            time=self.time.value,
            flux=self.flux.value,
            flux_err=self.flux_err.value,
            tstarts=self.flares["tstart"].values,
            tstops=self.flares["tstop"].values,
            buffer=buffer,
            max_flares=max_flares,
            plot=plot,
            debug_plot=debug_plot,
            **kwargs
        )
        self._flare_fit_results = results
        return self
        


    def flare_table(self, include_group_rows=False):
        """
        Build a summary table of fitted flare parameters.

        This method wraps around `make_flare_table()` and is intended to be called
        after running `flcd.fit_flares()`.

        Parameters
        ----------
        results : list
            Output from `fit_flares()`, containing fitted flare results.
        include_group_rows : bool
            If True, include rows for group-level fits in addition to individual flare components.
    
        Returns
        -------
        pandas.DataFrame
            Table with columns: t_peak, fwhm, amplitude, ed_rec, fit_type, group_index, etc.
        """
        
        return make_flare_table(self._flare_fit_results, include_group_rows=include_group_rows)



    def corner(self, index=0, param_names=None, **kwargs):
        """
        Plot corner plot aligned with the flare table row index.
    
        If the selected row is a group member, this function redirects to the group-level posterior.
        """
        if not hasattr(self, "_flare_fit_results"):
            raise ValueError("Run `fit_flares()` first before calling `corner()`.")
    
        results = self._flare_fit_results
    
        if index >= len(results):
            raise IndexError(f"Flare table has only {len(results)} entries, but index {index} was requested.")
    
        res = results[index]
    
        # Redirect to group fit if this is a group_member
        if res["fit_type"] == "group_member":
            group_idx = res["group_index"]
            group_result = next((r for r in results if r["fit_type"] == "group" and r["group_index"] == group_idx), None)
            if group_result is None:
                raise ValueError(f"Group fit for group_index={group_idx} not found.")
            res = group_result
    
        samples = res.get("posterior_samples")
        if samples is None:
            raise ValueError("No posterior samples available (likely not fitted with emcee).")
    
        nf = res["n_flares"]
        if param_names is None:
            param_names = [f"c{i}" for i in range(5)] + \
                          [f"tp{i}" for i in range(nf)] + \
                          [f"fwhm{i}" for i in range(nf)] + \
                          [f"amp{i}" for i in range(nf)]
    
        return plot_corner(samples, param_names=param_names, **kwargs)
