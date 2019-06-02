import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
name = 'altaipony'

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
LOG = logging.getLogger(__name__)


#LOG.addHandler(logging.StreamHandler())
# Note:
# logging.StreamHandler(stream=None): Returns a new instance of the StreamHandler
# class. If stream is specified, the instance will use it for logging output;
# otherwise, sys.stderr will be used.
