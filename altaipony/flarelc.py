from lightkurve import KeplerLightCurve, KeplerLightCurveFile, KeplerTargetPixelFile

class FlareLightCurve(KeplerLightCurve):
    """
    Flare light curve
    """
    def __init__(self, time=None, flux=None, flux_err=None, time_format=None, time_scale=None,
                 centroid_col=None, centroid_row=None, quality=None, quality_bitmask=None,
                 channel=None, campaign=None, quarter=None, mission=None, cadenceno=None,
                 targetid=None, ra=None, dec=None, label=None, meta={},
                 breakpoints=None, flares=None):

        super(FlareLightCurve, self).__init__(time=time, flux=flux, flux_err=flux_err, time_format=time_format, time_scale=time_scale,
                                              centroid_col=centroid_col, centroid_row=centroid_row, quality=quality,
                                              quality_bitmask=quality_bitmask, channel=channel, campaign=campaign, quarter=quarter,
                                              mission=mission, cadenceno=cadenceno, targetid=targetid, ra=ra, dec=dec, label=label,
                                              meta=meta)
        self.breakpoints = breakpoints
        self.flares = flares #pd.DataFrame(columns=['istart','istop','cstart','cstop', 'ed'])

    def __repr__(self):
        return('FlaresLightCurve(ID: {})'.format(self.targetid))

    def from_TPF(path, **kwargs):
        tpf = KeplerTargetPixelFile(path, **kwargs)
        lc = tpf.to_lightcurve()
        return from_KeplerLightCurve(lc)

    def from_KeplerLightCurveFile(path_or_ID):

        return KeplerLightCurveFile.from_archive(path_or_ID)

    @staticmethod
    def from_KeplerLightCurve(lc):


        return FlareLightCurve(time=lc.time, flux=lc.flux, )




    #IO
    # tpf from MAST or path
    # k2sc file from MAST or path
