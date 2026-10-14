
import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
name = 'altaipony'

import logging
# Show AltaiPony's own messages (e.g. how many flare candidates were found),
# without configuring the root logger: that would also print the INFO
# messages of every other library. Silence them with
# logging.getLogger("altaipony").setLevel(logging.WARNING).
LOG = logging.getLogger(__name__)
if not LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)


from .version import __version__


