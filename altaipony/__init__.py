import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
name = 'altaipony'

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
LOG = logging.getLogger(__name__)



