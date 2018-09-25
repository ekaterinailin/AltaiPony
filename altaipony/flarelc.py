from lightkurve import KeplerLightCurve, KeplerLightCurveFile

class FlareLightCurve(KeplerLightCurve):
    """
    Flare light curve
    """
    def __init__(self, time=None, flux=None, flux_err=None, time_format=None,
                 time_scale=None, targetid=None, label=None, meta={},
                 breakpoints=None, flares=None):

        super(KeplerLightCurve, self).__init__(self, time=None, flux=None,
                                               flux_err=None, time_format=None,
                                               time_scale=None, targetid=None,
                                               label=None, meta={})
        self.breakpoints =
        self.flares = pd.DataFrame(columns=['istart','istop','cstart','cstop',
                                             'ed'])
