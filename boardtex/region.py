# coding: utf-8

from __future__ import division, print_function

# Third-party
from matplotlib import pyplot, cm
import numpy as np
import scipy as sp

from skimage.data import imread
from skimage.color import rgb2grey
from skimage.measure import regionprops
from skimage.transform import resize


class NormalizedRegion(object):

    props = ['Area', 'ConvexArea', 'Eccentricity', 'EquivDiameter',
             'Extent', 'FilledArea', 'Orientation', 'Perimeter', 'Solidity']

    def __init__(self, image_data, shape=(64,64)):
        """ Given an array of image data around a single symbol, downsample to
            the given shape, convert to a binary image, and extract a 
            feature vector. 
            
            Parameters
            ----------
            image_data : array_like
                A 2D array of image data.
            shape : tuple (optional)
                Downsample the image to this size.
        """

        self.image = self._normalize(image_data, shape)
        self.rprops = self._regionprops()
        self.features = self._features()

    def _normalize(self, image, shape):
        """ Downsample the image and convert to binary """
        image = resize(image, shape)
        image = self._to_binary(image)
        return image

    def _to_binary(self, image, thresh=0.5, invert=False):
        """ Threshold an image and convert to binary """
        image = rgb2grey(image)
        if invert:
            return np.asarray(image < thresh, dtype='int')
        else:
            return np.asarray(image > thresh, dtype='int')

    @classmethod
    def from_file(self, filename, shape=(64,64)):
        image = imread(filename)
        # image = -image-1
        return NormalizedRegion(image, shape=shape)

    def _regionprops(self):
        regions = regionprops(self.image, properties=self.props)
        return regions[0]

    def _features(self):
        features = np.asarray(self.image.ravel(), dtype='float')
        extra_features = [self.rprops[prop] for prop in self.props]
        return np.append(features, extra_features)

    def show(self):
        pyplot.imshow(self.image, cmap=cm.Greys, interpolation='nearest')

def save_regions(regions, path="", prefix=None, ext="png"):
    """ Save a list of regions out to disk as individual images. 
    
        Parameters
        ----------
        regions : iterable of Region objects
        path : str (optional)
            Path to save the images to. Default is current working directory.
        prefix : str (optional)
            Prefix for each filename.
        ext : str (optional)
            Type of image to write, e.g., png, jpg, tif.
    """
    if prefix is None:
        filename_base = '{index}.{ext}' 
    else:
        filename_base = '{prefix}-{index}.{ext}'
        
    for index, region in enumerate(regions):
        sp.misc.imsave(filename_base.format(prefix=prefix, index=index, ext=ext), 
                       -region.image+1)

# class Region(object):
#     
#     def __init__(self, rdict, dim=64):
#         self.dim = dim
#         self.bounding_box = rdict['BoundingBox']
#         self.area = rdict['Area']
#         self.centroid = rdict['Centroid']
#         self.nrdict = self._normalize_region(rdict)
# 
#     def _normalize_region(rdict):
#         image = rdict['Image']
#         image = resize(image, (self.dim,self.dim))
#         image = to_binary(image)
#         props = [
#             'Image', 'Area', 'Centroid', 'ConvexArea', 'Eccentricity', 'EquivDiameter',
#             'Extent', 'FilledArea', 'Orientation', 'Perimeter', 'Solidity'
#         ]
#         regions = regionprops(image, properties=props)
#         return regions[1]
# 
#     @property
#     def image(self):
#         return self.nrdict['Image']
# 
#     @property
#     def features(self):
# 
#     def show(self):
#         pyplot.imshow(self.image, cmap=cm.Greys)
# 
#     def to_file(self, filename):
#         sp.misc.imsave(filename, -self.image+1)
