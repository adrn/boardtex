# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob

# Third-party
import numpy as np
import pytest

from ..region import NormalizedRegion, save_regions
from ..split import split

def test_save_regions():
    regions = split("data/basic.jpg", shape=(16,16))
    save_regions(regions, path="/tmp/", prefix="basic")