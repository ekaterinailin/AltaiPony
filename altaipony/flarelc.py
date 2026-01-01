import numpy as np
import pandas as pd
import os
import copy
import logging
import progressbar
import datetime
import warnings


from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
        

from .flares import flare_factor
from .fit_flares import fit_flares as fit_flares_func
from .fit_flares import make_flare_table
from .fit_flares import plot_corner

from lightkurve import LightCurve
from lightkurve.utils import KeplerQualityFlags


from .altai import (find_flares,
                    find_iterative_median, 
                    detrend_savgol)
from .fakeflares import (merge_fake_and_recovered_events,
                         generate_fake_flare_distribution,
                         mod_random,
                         flare_model,
                         )
from .injrecanalysis import wrap_characterization_of_flares, _heatmap
from .utils import split_gaps
from .utils import get_response_curve

import time
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
        K2SC detrend flux, same units as flux.
    detrended_flux_err : array-like
        K2SC detrend flux error, same units as flux.
    gaps : list of tuples of ints
        Each tuple contains the start and end indices of observation gaps. See
        ``find_gaps``.
    flares : DataFrame
        Table of flares, their start and stop time, recovered equivalent duration
        (ED), and, if applicable, recovery probability, ratio of recovered ED to
        injected synthetic ED. Also information about quality flags may be stored
        here.
    it_med : array-like
        Iterative median, see the ``find_iterative_median`` method.


    """


    @property
    def detrended_flux_err(self) -> np.array:
        try:
            return self["detrended_flux_err"]
        except KeyError:
            self["detrended_flux_err"] = np.full_like(self.time.value, np.nan)
            return self["detrended_flux_err"]

    @detrended_flux_err.setter
    def detrended_flux_err(self, detrended_flux_err):
        self["detrended_flux_err"] = detrended_flux_err


    @property
    def detrended_flux(self) -> np.array:
        try:
            return self["detrended_flux"]
        except KeyError:
            self["detrended_flux"] = np.full_like(self.time.value, np.nan)
            return self["detrended_flux"]

    @detrended_flux.setter
    def detrended_flux(self, detrended_flux):
        self["detrended_flux"] = detrended_flux 


    @property
    def it_med(self):
        try:
            return self["it_med"]
        except KeyError:
            self["it_med"] = np.full_like(self.time.value, np.nan)
            return self["it_med"]

    @it_med.setter
    def it_med(self, it_med):
        self["it_med"] = it_med

    @property
    def origin(self):
        try:
            return self.meta["origin"]
        except KeyError:
            self.meta["origin"] = "FLC"
            return self.meta["origin"]

    @origin.setter
    def origin(self, origin):
        self.meta["origin"] = origin 



    @property
    def flares(self) -> pd.DataFrame:
        try:
            return self.meta["flares"]
        except KeyError:
            self.meta["flares"] = pd.DataFrame(columns=FLARE_COLUMNS)
            return self.meta["flares"]

    @flares.setter
    def flares(self, flares):
        self.meta["flares"] = flares 


    @property
    def fake_flares(self) -> pd.DataFrame:
        
        try:
            return self.meta["fake_flares"]
        except KeyError:
            self.meta["fake_flares"] = pd.DataFrame(columns=FAKE_FLARE_COLUMNS)
            return self.meta["fake_flares"]

    @fake_flares.setter
    def fake_flares(self, fake_flares):
        self.meta["fake_flares"] = fake_flares


    @property
    def gaps(self):
        try:
            return self.meta["gaps"]
        except KeyError:
            self.meta["gaps"] = None
            return self.meta["gaps"]

    @gaps.setter
    def gaps(self, gaps):
        self.meta["gaps"] = gaps 

    @property
    def cadenceno(self) -> np.array:
        """Cadence number for each observation."""
        try:
            return self["cadenceno"]
        except KeyError:
            # Initialize with sequential integers if not present
            self["cadenceno"] = np.arange(len(self.time), dtype=int)
            return self["cadenceno"]

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
        
        return(f'FlareLightCurve(ID: {self.targetid:<9} |' \
            f' Mission: {mission:<6} |' \
            f' QCS: {qcs:>3} |' \
            f' Cadence: {self.meta.get("cadence", "Unknown"):.0f} s')

    
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
                f"{self.meta.get('cadence','Unknown'):.0f}.fits"
        
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
        
        # Convert time to Time object with proper format
        # Check what format/unit the time has
        if hasattr(time_array, 'unit'):
            time_unit_str = str(time_array.unit)
            if time_unit_str in ['bkjd', 'd', 'day']:
                # Time in days - convert to Time object with JD format
                time_data = Time(time_array.value, format='jd')
            else:
                # Try to create Time object directly
                time_data = Time(time_array.value, format='jd')
        else:
            # No unit, assume JD
            time_data = Time(time_array, format='jd')
        
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
            raise ValueError("TARGETID not found in FITS header.")
        
        return flc
    
    def find_gaps(self, maxgap=0.09, minspan=10, splits=[]):
        '''
        Find gaps in light curve and stores them in the gaps attribute.
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

        '''
        lc = copy.copy(self)
        dt = np.diff(lc.time.value)
        gap = np.where(np.append(0, dt) >= maxgap)[0]

        # add start/end of LC to loop over easily
        gap_out = np.append(0, np.append(gap, len(lc.time.value)))

        # left start, right end of data
        left, right = gap_out[:-1], gap_out[1:]

        # drop too short observation periods
        too_short = np.where(np.diff(gap_out) < 10)
        left, right = np.delete(left,too_short), np.delete(right,(too_short))

        # get the gaps
        gaps = list(zip(left, right))
        
        # split up the time series in additional place if needed
        lc.gaps = split_gaps(gaps, splits)

        return lc

    def detrend(self, mode, save=False,
                path='detrended_lc.fits', de_niter=30, max_sigma=3, 
                func=None,
                **kwargs):
        """
        De-trends a FlareLightCurve using ``K2SC``.
        Optionally saves the LightCurve in a fits file that can
        be read as K2SC file.

        Parameters:
        ----------
        mode : str
            "savgol" or "custom"
        de_niter : int
            Differential Evolution global optimizer parameter. K2SC
            default is 150, here set to 30 as a safety net to avoid
            unintenional computational effort.
        max_sigma: int
            Default is 3, value is passed to iterative sigma clipping
            in K2SC
        save : False or bool
            If True, the light curve is saved as a fits file to a
            given folder.
        path : str
            Path to resulting fits file. 
            As a default, the fits file will be stored in the
            working directory.
        func : function
            custom detrending function
        kwargs : dict
            Keyword arguments to pass to k2sc, detrend_savgol, or custom
            method

        Returns
        --------
        FlareLightCurve
        """
        if mode == "savgol":
        
            new_lc = copy.deepcopy(self)
            new_lc =  detrend_savgol(new_lc, **kwargs)
            if save == True:
                new_lc.to_fits(path)
            return new_lc
        

        elif mode=="custom":
            
            if func is None:
                LOG.error('If you wish to use a custom detrending function you'
                          ' must pass a callable to the "func" parameter.')
                raise ValueError

            new_lc = copy.deepcopy(self)
            
            new_lc = func(new_lc, **kwargs)            
            
            if (np.isnan(new_lc.detrended_flux).all() | np.isnan(new_lc.detrended_flux_err).all()):
                LOG.error('The custom de-trending function you passed does not'
                          ' return an detrended_flux or detrended_flux_err attri'
                          'bute.')
                raise AttributeError
            
            if save == True:
                new_lc.to_fits(path)
            
            return new_lc
        
        else:
            err_str = (f'\nDe-trending mode {mode} does not exist. Pass "savgol" (for a Savitzky-Golay filter based detrending)'
                       ' or "custom" (to pass a custom detrending function to func=).')
            LOG.exception(err_str)
            raise ValueError(err_str)


            
            
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


        
            
            
            
            
            
    def find_flares(self, minsep=3, fake=False, **kwargs):

        '''
        Find flares in a ``FlareLightCurve``.

        Parameters
        -------------
        minsep : 3 or int
            Minimum distance between two candidate start times in datapoints.
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
        
        Returns
        ----------
        FlareLightCurve
        '''
        if ((fake==False) & (self.flares.shape[0]>0)):
            return self
        else:
            lc = copy.deepcopy(self)
            #re-init flares
            columns = ['istart', 'istop', 'cstart', 'cstop', 'tstart',
                       'tstop', 'ed_rec', 'ed_rec_err', 'ampl_rec', 'dur']
            lc.flares = pd.DataFrame(columns=columns)
            #find continuous observing periods
            lc = lc.find_gaps()
            #find the true median value iteratively
            lc = find_iterative_median(lc)
            #find flares
            lc = find_flares(lc, minsep=minsep, **kwargs)
            

        return lc

    def sample_flare_recovery(self, iterations=2000, inject_before_detrending=False,
                              mode=None, func=None, save_lc_to_file=False, folder="", 
                              fakefreq=0.05, save=False, path=None, detrend_kwargs={},
                              **kwargs):
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
        mode : str
            "savgol" or "k2sc". Required if ``inject_before_detrending`` is True.
        fakefreq : 0.05 or float
            number of flares per day, but at least one per continuous observation period will be injected
        detrend_kwargs : dict
            Keyword arguments to pass to FlareLightCurve.detrend
        kwargs : dict
            Keyword arguments to pass to inject_fake_flares

        Returns
        -------
        lc : FlareLightCurve
            Detrended LC with all fake_flares listed in the attribute
        fake_lc : FlareLightCurve
            Light curve with the last iteration of synthetic flares injected.
        """
        injrecstr = {True : "before", False : "after"} # define string to identify fake flare analysis by file name
        
        lc = copy.deepcopy(self)
        if inject_before_detrending == True:
            lc = lc.detrend(mode, func=func, **detrend_kwargs)
        lc = lc.find_gaps()
        lc = lc.find_flares()
        lc = find_iterative_median(lc)
        
        lc_ = copy.deepcopy(lc)
        
        columns =  ['istart', 'istop', 'cstart', 'cstop', 'tstart', 'tstop',
                    'ed_rec', 'ed_rec_err', 'duration_d', 'amplitude', 'ed_inj',
                    'peak_time', 'ampl_rec', 'dur']
        

        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=iterations).start()
        for i in range(iterations):
            fake_lc = lc.inject_fake_flares(inject_before_detrending=inject_before_detrending,
                                                 fakefreq=fakefreq,
                                                 **kwargs)
            
            if save_lc_to_file == True:
                fake_lc.to_fits("{folder}before.fits")
                print(f"saved {self.targetit} LC before detrending")
                
            injs = fake_lc.fake_flares
           
            if inject_before_detrending == True:
                LOG.info('\nDetrending fake LC:\n')
                fake_lc = fake_lc.detrend(mode, func=func, **detrend_kwargs)
            
            fake_lc = fake_lc.find_flares(fake=True)
            recs = fake_lc.flares
          
            if save_lc_to_file == True:
                fake_lc.to_fits(f"{folder}after.fits")
                print(f"saved {self.targetit} LC after detrending")
                
            # The following line makes sure that column order is retained    
            injrec_results = pd.DataFrame(columns=columns)
            
            # Merge injected and recovered flares
            injrec_results = pd.concat([injrec_results, merge_fake_and_recovered_events(injs, recs)],
                                                   ignore_index=True)
            

            bar.update(i + 1)
            
            if save == True:
            
                #Define default path if needed
                if path is None:
                    path = (f'{iterations}_{lc.targetid}_inj_'
                            f'{injrecstr[inject_before_detrending]}_'
                            f'{lc.campaign}.csv')
                            
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

        
        # End monitoring
        bar.finish()
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
        FlareLightCurve with the flares table supplemented with an integer
        ``quality`` and, if applicable, a string ``explanation`` column.
        """
        lc = copy.copy(self)
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

    def get_saturation(self, factor=10, return_level=False):
        """
        Goes back to the TPF and measures the maximum saturation level during a
        flare, averaged over the aperture mask.

        Parameters
        -----------
        factor : 10 or float
            Saturation level in full well depths.

        Returns
        -------
        FlareLightCurve with modified 'flares' attribute.
        """
        flc = copy.copy(self)
        well_depth = 10093

        def sat(flares, flc=flc, well_depth=10093, return_level=False):
            pfl = flc.pixel_flux[flares.istart:flares.istop]
            flare_aperture_pfl = pfl[:,flc.pipeline_mask]
            return sat_level(flare_aperture_pfl, well_depth, return_level)
            
        def sat_level(flare_aperture_pfl, well_depth, return_level):
            saturation_level = np.nanmean(flare_aperture_pfl, axis=1) / well_depth
            if return_level == False:
                return np.any(saturation_level > factor)
            else:
                return np.nanmax(saturation_level)
        
        colname = 'saturation_f{}'.format(factor)
        
        if np.isnan(flc.saturation).all():
            
            if flc.flares.shape[0] > 0:#do not attempt if no flares are detected
                flc.flares[colname] = flc.flares.apply(sat, axis=1,
                                                    return_level=return_level)
                
            elif flc.flares.shape[0] == 0: # calculate saturation for all times
                flare_aperture_pfl = flc.pixel_flux[:,flc.pipeline_mask]
                saturation_level = np.nanmax(flare_aperture_pfl, axis=tuple(np.arange(len(flare_aperture_pfl.shape)))[1:]) / well_depth
                if return_level == False:
                    flc.saturation = saturation_level > factor
                else:
                    flc.saturation = saturation_level
                                           

        else:
             if flc.flares.shape[0] > 0:#do not attempt if no flares are detected
                 if isinstance(flc.saturation[0], np.bool_) :
                     if return_level == False:
                        flc.flares[colname] = flc.flares.apply(lambda x: (flc.saturation[x.istart: x.istop] == True).any(),
                                                        axis=1)
                     elif return_level == True:
                         LOG.info("Warning: Saturation is given as boolean flag. \n Choose return_level=False.")
                         return flc.get_saturation(factor=factor, return_level=False)
                    
                 elif (isinstance(flc.saturation[0], np.float64) | isinstance(flc.saturation[0], np.float32)) :
                     if return_level == False:
                        flc.flares[colname] = flc.flares.apply(lambda x: (flc.saturation[x.istart: x.istop] > factor).any(),
                                                               axis=1)

                     elif return_level == True:
                        flc.flares[colname] = flc.flares.apply(lambda x: np.nanmax(flc.saturation[x.istart: x.istop]),
                                                        axis=1)

                                               

        return flc


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

        def _equivalent_duration(time, flux):
            '''
            Compute the Equivalent Duration of a fake flare.
            This is the area under the flare, in relative flux units.

            Parameters:
            -------------
            time : numpy array
                units of DAYS
            flux : numpy array
                relative flux units
            Return:
            ------------
            p : float
                equivalent duration of a single event in units of seconds
            '''
            x = time * 60.0 * 60.0 * 24.0
            integral = np.sum(np.diff(x) * flux[:-1])
            return integral
        
        
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
        # At least one per gap
        # or as defined by the frequency
        nfakesum = max(len(fake_lc.gaps),
                       int(np.rint(fakefreq *
                           (fake_lc.time.value.max() - fake_lc.time.value.min()))
                           )
                       )
        
        # Use a light curve where you know the median flux
        fake_lc = find_iterative_median(fake_lc)
        
        # Init arrays for the synthetic flare parameters
        t0_fake = np.zeros(nfakesum, dtype='float') # peak times
        ed_fake = np.zeros(nfakesum, dtype='float') # ED
        dur_fake = np.zeros(nfakesum, dtype='float') # duration
        ampl_fake = np.zeros(nfakesum, dtype='float') # amplitude
        
        # Init the synthetic flare counter to allow to point to the right
        # places in the arrays above (XXX_fake etc.)
        ckm = 0
        
        # Iterate over continuous observing periods
        for (le,ri) in fake_lc.gaps:
            
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
                    ed_fake[k] = _equivalent_duration(time, fl_flux)
                    
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
        corrected flare characteristics.
        
        """
        flc = copy.deepcopy(self)
        flares = wrap_characterization_of_flares(flc.fake_flares, flc.flares,
                                                 flares_per_bin=flares_per_bin,
                                                 ampl_bins=ampl_bins,
                                                 dur_bins=dur_bins)
        flc.flares = flares
        return flc
    


            
    def fit_flares(self, method="emcee", buffer=0.05, max_flares=3, delta_bic=0.0, plot=True, debug_plot=False, **kwargs):
        """
        Fit flares using a polynomial baseline + analytic flare model (Davenport et al. 2014).

        This method automatically uses:
        - time
        - flux and flux_err
        - 'tstart' and 'tstop' as fitting regions

        Parameters
        ----------
        method : str
            One of "curve_fit" or "emcee" (default: "curve_fit").
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
        results : list
            A list of fitted model results.
        """
        results = fit_flares_func(
            time=self.time.value,
            flux=self.flux.value,
            flux_err=self.flux_err.value,
            tstarts=self.flares["tstart"].values,
            tstops=self.flares["tstop"].values,
            method=method,
            buffer=buffer,
            max_flares=max_flares,
            plot=plot,
            debug_plot=debug_plot,
            **kwargs
        )
        self._flare_fit_results = results
        return results


    def flare_table(self, results, include_group_rows=False):
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
        
        return make_flare_table(results, include_group_rows=include_group_rows)



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



def generate_lightcurve(errorval,  a1, a2, period1, period2, quad, cube,
                        mean=3400.):
    
    """Generate wild light curves with variability on several
    timescales.
    
    Returns:
    ---------
    FlareLightCurve with time, flux, and flux_err attributes
    """
    time = np.arange(10, 10 + 10 * np.pi,.0008)

    # define the flux
    flux = (np.random.normal(0,errorval,time.shape[0]) +
            mean + 
            a1*mean*np.sin(period1*time +1.)  +
            a2*mean*np.sin(period2*time) +
            quad*(time-25)**2 -
            cube*(time-25)**3)

    # add a gap in the data
    flux[5600:7720] = np.nan

    # add big and long flare
    l = 66
    flux[5280:5280 + l] = flux[5280:5280 + l] + np.linspace(1000,250,l)

    # add tiny flare
    l = 3
    flux[15280:15280 + l] = flux[15280:15280 + l] + np.linspace(100,60,l)

    # add intermediate flare
    l, s = 15, 25280
    flux[s:s + l] = flux[s:s + l] + np.linspace(200,60,l)

    # typically Kepler and TESS underestimate the real noise
    err = np.full_like(time,errorval/3*2)

    # define FLC
    return FlareLightCurve(time=time, flux=flux, flux_err=err)



