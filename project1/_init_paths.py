"""Set up paths for Project1."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add scikit-learn to PYTHONPATH
lib_path = osp.join(this_dir, 'scikit-learn')
add_path(lib_path)
