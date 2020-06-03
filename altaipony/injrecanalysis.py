import pandas as pd
import numpy as np

import seaborn
import matplotlib.pyplot as plt

import copy

def wrap_characterization_of_flares(injrec, flares, ampl_bins=70, dur_bins=160):
    """Take injection-recovery results for a data set
    and the corresponding flare table. Determine
    recovery probability, ED ratio, amplitude ratio,
    duration ratio, and the respective standard deviation.
    Count on how many synthetic flares the results are based.

    Parameters:
    -----------
    injrec : DataFrame
        table with injection-recovery results from AltaiPony
    flares : DataFrame
        table with flare candidates detected by AltaiPony
    ampl_bins : int
        number of bins in amplitude
    dur_bins : int
        number of bins in duration

    Return:
    --------
    DataFrame : flares and injrec merged with the characteristics
                listed above.
    """

    flares = flares.dropna(subset=["ed_rec"])
    injrec.ed_rec = injrec.ed_rec.fillna(0)
    injrec['rec'] = injrec.ed_rec.astype(bool).astype(float)
    injrec['dur'] = injrec.tstop - injrec.tstart
    flares['dur'] = flares.tstop - flares.tstart

    ampl_bins = np.linspace(min(injrec.ampl_rec.min(),
                                flares.ampl_rec.min(), 
                                injrec.amplitude.min()),
                            max(injrec.ampl_rec.max(),
                                flares.ampl_rec.max(),
                                injrec.amplitude.max()), 
                            ampl_bins)

    dur_bins = np.linspace(min(injrec.dur.min(),
                               flares.dur.min(), 
                               injrec.duration_d.min()),
                           max(injrec.dur.max(),
                               flares.dur.max(),
                               injrec.duration_d.max()), 
                           dur_bins)

    flcc, dscc = characterize_flares(flares, injrec, otherfunc="count",
                            amplrec="ampl_rec", durrec="dur",
                            amplinj="amplitude", durinj="duration_d",
                            ampl_bins=ampl_bins,
                            dur_bins=dur_bins)
    fl, ds = characterize_flares(flares, injrec, otherfunc="std",
                            amplrec="ampl_rec", durrec="dur",
                            amplinj="amplitude", durinj="duration_d",
                            ampl_bins=ampl_bins,
                            dur_bins=dur_bins)
    fl = fl.merge(flcc)
    fl["ed_corr_err"] = np.sqrt(fl.ed_rec_err**2 + fl.ed_corr**2 * fl.ed_ratio_std**2)
    fl["amplitude_corr_err"] = fl.amplitude_corr * fl.amplitude_ratio_std / fl.amplitude_ratio
    fl["duration_corr_err"] = fl.duration_corr * fl.duration_ratio_std / fl.duration_ratio
    return fl


def characterize_flares(flares, df, otherfunc="count",
                        amplrec="ampl_rec", durrec="dur",
                        amplinj="amplitude", durinj="duration_d",
                        **kwargs):
    """Assign ED recovery ratios, and
    recovery probability to all flares
    whose recovered parameters are covered
    by the synthetic data.

    Parameters:
    -----------
    flares : DataFrame
        flare table
    df : DataFrame
        injection-recovery table
    otherfunc : str
        additional column for statistical analysis.
        Can accept "count", "std", and other simple
        pandas methods that work on Groupby objects.
    amplrec : str
        column name for recovered amplitude
    durrec : str
        column name for recovered duration
    amplinj: str
        column name for injected amplitude
    durrec : str
        column name for injected duration
    kwargs : dict
        Keyword arguments to pass to tile_up_injection_recovery

    Return:
    -------
    DataFrame: flares with additional columns
    """
    # define observed flare duration
    flares["dur"] = flares.tstop - flares.tstart
    ds =dict()

    # calculate inj-rec ratio for ED, amplitude, and duration
    for typ in ["ed_ratio", "amplitude_ratio", "duration_ratio"]:
        d, val = tile_up_injection_recovery(df, typ, otherfunc=otherfunc,
                                            ampl=amplrec, dur=durrec, **kwargs)
        d = d.dropna(how="all", axis=0)
        ds[typ] = d
        helper = lambda x: multiindex_into_df_with_nans(x, d,
                                                        i1="ampl_rec",
                                                        i2="dur", i3=val)
        flares[typ] = flares.apply(helper, axis=1)
        helper = lambda x: multiindex_into_df_with_nans(x, d,
                                                        i1="ampl_rec",
                                                        i2="dur", i3=otherfunc)
        flares["{}_{}".format(typ, otherfunc)] = flares.apply(helper, axis=1)

    # calculate recovery probability from corrected values
    flares["amplitude_corr"] = flares[amplrec] / flares.amplitude_ratio
    flares["duration_corr"] = flares[durrec] / flares.duration_ratio
    flares["ed_corr"] = flares["ed_rec"] / flares.ed_ratio
    d, val = tile_up_injection_recovery(df, "recovery_probability",
                                        otherfunc=otherfunc, ampl=amplinj,
                                        dur=durinj, **kwargs)
    d = d.dropna(how="all", axis=0)
    ds["recovery_probability"] = d
    helper = lambda x: multiindex_into_df_with_nans(x, d,
                                                    i1="amplitude_corr",
                                                    i2="duration_corr", i3=val)
    flares["recovery_probability"] = flares.apply(helper, axis=1)
    helper = lambda x: multiindex_into_df_with_nans(x, d,
                                                    i1="amplitude_corr",
                                                    i2="duration_corr", i3=otherfunc)
    flares["{}_{}".format("recovery_probability", otherfunc)] = flares.apply(helper, axis=1)

    return flares, ds


def tile_up_injection_recovery(df, typ, ampl="amplitude", dur="duration_d",
                               otherfunc = "count",
                               ampl_bins=np.arange(0, .5, 0.025),
                               dur_bins=np.arange(0, .2, 5e-3)):
    """Tile up the injection recovery data into
    amplitude and duration bins. Return a multiindexed
    matrix that can be accessed to assign recovered
    ED/amplitude/duration ratio or recovery probability
    to a given observation (AMPL, DUR)
    or its recovery corrected form.

    Parameters:
    ------------
    df : DataFrame
        injection recovery table
    typ: str
        type of inj-rec parameter to obtain
        Can be "recovery_probability",
               "ed_ratio",
               "amplitude_ratio",
               "duration_ratio".
    ampl: str
        column name used to bin on one parameter axis
    dur : str
        column name used to bin on the other axis
    otherfunc : pandas groupby applicable function string
        "std", "count", "mean" ...
        Use this to get another statistic on the desired
        inj-rec parameter that is not median
    ampl_bins : numpy array
        bins for one axis, should cover both
        injected and recovered range
    dur_bins : numpy array
        bins for the other axis, should cover both
        injected and recovered range

    Return:
    -------

    multiindexed DataFrame, str :
        tiled injection-recovery dataset,
        column name for relevant parameter
    """

    # Calculate helpful columns
    if "rec" not in df.columns:
        df["rec"] = df.ed_rec.fillna(0).astype(bool).astype(int)
    if "dur" not in df.columns:
        df["dur"] = df.tstop - df.tstart
    
    
    d1 = df.assign(Amplitude=pd.cut(df[ampl], ampl_bins),
                   Duration=pd.cut(df[dur],  dur_bins))

    types = {"ed_ratio":("ed_rec","ed_inj","edrat"),
             "amplitude_ratio":("ampl_rec","amplitude","amplrat"),
             "duration_ratio":("dur","duration_d","durrat"),
            }

    if typ == "recovery_probability":
        grouped = d1.groupby(["Amplitude","Duration"])
        d2 = grouped.rec.sum() / grouped.rec.count()
        d3 = getattr(grouped.rec, otherfunc)()
        val = "rec"

    else:
        d1["rel"] = d1[types[typ][0]] / d1[types[typ][1]]
        grouped = d1.groupby(["Amplitude","Duration"])
        d2 = grouped.rel.median()
        d3 = getattr(grouped.rel, otherfunc)()
        val = types[typ][2]

    return pd.DataFrame({val : d2, otherfunc : d3}), val


def multiindex_into_df_with_nans(x, df, i1="ampl_rec", i2="dur", i3="edrat"):
    """Helps with indexing in multiindexed tables
    that also have NaNs.

    Parameter:
    ---------
    x : Series
        row from the flare detection table
    df : DataFrame
        multiindexed table with NaNs
    i1, i2, i3: str, str, str
        name of 1st index, 2nd index and value column
        in df
    Return:
    -------
    float : value at index given by x
    """
    try:
        return df.loc[(x[i1], x[i2]), i3]
    except KeyError:
        return np.nan


def percentile(x, q):
    """Calculate percentile q in Series x.

    Parameters:
    ------------
    x : pandas Series
        distribution
    q : float
        desired percentile (0,100)

    Return:
    --------
    float
    """
    if (np.isnan(x.values).all() | np.isnan(q)):
        return np.nan
    else:
        return np.percentile(x.dropna(), q=q)

def _heatmap(flcd, typ, ampl_bins, dur_bins, flares_per_bin):
    """Create a heatmap for either recovery probability or ED ratio.
    
    Parameters:
    -----------
    typ : string
        Either "recovery_probability" or "ed_ratio"
    ampl_bins : int or array
        bins for amplitudes
    dur_bins : int or array
        bins for duration or FWHM
    flares_per_bin : int
        number of flares per bin
    """

    if not bool(flcd.fake_flares.shape[0] > 0):
        raise AttributeError("Missing injection-recovery data. "
                             "Use `FLC.load_injrec_data(path)` to fetch "
                             "some, or run `FLC.sample_flare_recovery()`.")
    
    # Did the use give appropriate bins?
    bins = np.array([bool(ampl_bins is not None),bool(dur_bins is not None)])
    
    # If only one out of [ampl_bins, dur_bins] is specified
    # specify the other by fixing the `flares_per_bin`
    if ((bins.any()) & (~bins.all())):
        
        # Which one is not defined?
        if ampl_bins is None:
            b = copy.copy(dur_bins)
        elif dur_bins is None:
            b = copy.copy(ampl_bins)
            
        # If defined bins are given as array, find length
        if (isinstance(b, float) | isinstance(b, int)):
            l = b
        else:
            l = len(b)    

        # Define the other bins accordingly
        if ampl_bins is None:
            ampl_bins = int(np.rint(flcd.fake_flares.shape[0] / l / flares_per_bin))
        elif dur_bins is None:
            dur_bins = int(np.rint(flcd.fake_flares.shape[0] / l / flares_per_bin))
    
    # If no bins are specified, choose bins of equal size
    # with approximately `flares_per_bin` in each bin:
    elif ~bins.any():
        bins = int(np.rint(np.sqrt(flcd.fake_flares.shape[0] / flares_per_bin)))
        ampl_bins, dur_bins = bins, bins
   
    # Tile up the inj-rec table using the bins.
    dff, val = tile_up_injection_recovery(flcd.fake_flares, 
                                          typ,
                                          ampl_bins=ampl_bins,
                                          dur_bins=dur_bins,)
    
    # Map internal keywords to human-readable ones:
    typ_map = {"recovery_probability" : 
               ["injected", "FWHM", "recovery probability"],
               "ed_ratio" : 
               ["recovered", "duration", "ED ratio"]}

    # Create a heatmap
    fig = plot_heatmap(dff, val, ID=flcd.targetid, label=typ_map[typ][2],
                       ylabel=f"{typ_map[typ][0]} amplitude", 
                       xlabel=f"{typ_map[typ][0]} {typ_map[typ][1]} [d]");
    
    return
def plot_heatmap(df, val, label=None,
                 ID=None, valcbr=(0.,1.),
                 ovalcbr=(0,50), xlabel="duration [d]",
                 ylabel="amplitude", cmap="viridis",
                 font_scale=1.5):
    """Plot a heatmap from the "fake_flares" table. 
    
    Parameters:
    ------------
    df : DataFrame
        fake_flares attribute or equivalent table
    val : str
        column name in df to map
    label: str
        human-readable version of "val"
    ID : int or str
        target id
    valcbr : tuple
        value range for "val"
    xlabel : str or "duration [d]"
        xlabel for plot
    ylabel : str or "amplitude"
        ylabel for plot   
    cmap : colormap
        default "viridis"
    font_scale : float
        set the size of tick labels, and bar label
    
    Return:
    -------
    matplotlib.figure.Figure        
    """

    # configure Seaborn
    seaborn.set(font_scale=font_scale)

    # Find the midpoint of the interval to use as ticks
    df = df.reset_index()
    df.Amplitude = df.Amplitude.apply(lambda x: x.mid)
    df.Duration = df.Duration.apply(lambda x: x.mid)
    
    # Init figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (9,7))
    
    # Some layout stuff
    if label is None:
        label = val
    
    # Create heatmap data format 
    heatmap1_data = pd.pivot_table(df, values=val, 
                         index=['Amplitude'], 
                         columns=['Duration'])

    try:
        heatmap = seaborn.heatmap(heatmap1_data, cmap=cmap,cbar_kws={'label': label},
                              vmin=valcbr[0], vmax=valcbr[1], annot=False, ax=ax,
                              yticklabels=["{:.2f}".format(x) for x in heatmap1_data.index.values],
                              xticklabels=["{:.3f}".format(x) for x in heatmap1_data.columns.values])
    except AttributeError:
        heatmap = seaborn.heatmap(heatmap1_data, cmap=cmap,cbar_kws={'label': label},
                              vmin=valcbr[0], vmax=valcbr[1], annot=False, ax=ax,
                              yticklabels=["{:.2f}".format(x) for x in heatmap1_data.index.values.categories.values.mid.values],
                              xticklabels=["{:.3f}".format(x) for x in heatmap1_data.columns.values.categories.values.mid.values])
    
    fig = heatmap.get_figure()
    
    # Do some layout stuff
    
    fig.tight_layout()
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)    
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(ID, fontsize=16)

    return fig
