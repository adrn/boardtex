# coding: utf-8

from __future__ import division, print_function

# Standard library
import os

# Third-party
import numpy as np
import scipy as sp

import Image
from skimage.color import rgb2grey
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.transform import resize

from .region import NormalizedRegion

def to_binary(image, thresh=0.5, invert=True):
    image = rgb2grey(image)
    if invert:
        return np.asarray(image < thresh, dtype='int')
    else:
        return np.asarray(image > thresh, dtype='int')    

def split(image, shape=(64,64)):
    """ Return a list of NormalizedRegion objects from a composite image. """
    
    if isinstance(image, basestring):
        image = imread(image)
    
    bin_image = to_binary(image)
    clear_image = clear_border(bin_image)
    # We need the +1 to properly offset the labels for regionprops
    label_image = label(clear_image, background=0)+1
    props = [
        'Image', 'BoundingBox', 'Centroid', 'Area',
    ]
    regions = regionprops(label_image, properties=props)
    regions = [NormalizedRegion(region['Image'], shape=shape) for region in regions]
    return regions
