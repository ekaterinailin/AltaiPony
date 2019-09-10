import os
from ..__init__ import PACKAGEDIR

test_ids = [211119999, 210951703, 211117077]
test_paths = [os.path.join(PACKAGEDIR, 'examples',
              'hlsp_k2sc_k2_llc_{}-c04_kepler_v2_lc.fits'.format(id_))
              for id_ in test_ids]
kepler_path = os.path.join(PACKAGEDIR, 'examples',
                           'kplr010002792-2009259160929_llc.fits')
                           
pathkepler = "altaipony/examples/kplr010002792-2009259160929_llc.fits"
pathk2LC = "altaipony/examples/ktwo211117077-c04_llc.fits"
pathk2TPF = "altaipony/examples/ktwo210994964-c04_lpd-targ.fits"
pathtess = "altaipony/examples/tess2018206045859-s0001-0000000358108509-0120-s_lc.fits"
pathAltaiPony = "altaipony/examples/pony010002792-2009259160929_llc.fits"

