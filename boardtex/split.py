import numpy as np
import scipy as sp

from skimage.data import imread
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
        

def split(image, dim=64):
    """Return a list of NormalizedRegion objects from a composite image."""
    bin_image = to_binary(image)
    clear_image = clear_border(bin_image)
    # We need the +1 to properly offset the labels for regionprops
    label_image = label(clear_image, background=0)+1
    props = [
        'Image', 'BoundingBox', 'Centroid', 'Area',
    ]
    regions = regionprops(label_image, properties=props)
    regions = [NormalizedRegion(region['Image'], dim=dim) for region in regions]
    return regions
