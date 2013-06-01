import numpy as np
import scipy as sp

from skimage.data import imread
from skimage.color import rgb2grey
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.transform import resize

def to_binary(image, thresh=0.5, invert=True):
    image = rgb2grey(image)
    if invert:
        return np.asarray(image < thresh, dtype='int')
    else:
        return np.asarray(image > thresh, dtype='int')
        

def split(image):
    clear_image = clear_border(image)
    # We need the +1 to properly offset the labels for regionprops
    label_image = label(clear_image, background=0)+1
    props = [
        'Image', 'BoundingBox', 'Centroid', 'Area',
    ]
    regions = regionprops(label_image, properties=props)
    return regions

def normalize_region(region, dim=64):
    image = region['Image']
    image = resize(image, (dim,dim))
    image = to_binary(image)
    props = [
        'Image', 'Area', 'Centroid', 'ConvexArea', 'Eccentricity', 'EquivDiameter',
        'Extent', 'FilledArea', 'Orientation', 'Perimeter', 'Solidity'
    ]
    regions = regionprops(image, properties=props)
    return regions[1]

def save_regions(regions, prefix):
    for index, region in enumerate(regions):
        sp.misc.imsave('%s-%s.jpg' % (prefix, index), -region['Image']+1)


    
    

