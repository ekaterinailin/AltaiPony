import pandas as pd
import numpy as np



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

    #
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
