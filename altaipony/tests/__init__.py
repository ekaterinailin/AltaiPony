import os
from ..__init__ import PACKAGEDIR

test_ids = ['211119999', '210951703', 211117077]
test_paths = [os.path.join(PACKAGEDIR, 'examples',
              'hlsp_k2sc_k2_llc_{}-c04_kepler_v2_lc.fits'.format(id_))
              for id_ in test_ids]
