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

    props = [
            'Area', 'ConvexArea', 'Eccentricity', 'EquivDiameter',
            'Extent', 'FilledArea', 'Orientation', 'Perimeter', 'Solidity'
    ]

    def __init__(self, image, dim=64):
        self.dim = dim
        self.image = self._normalize(image)
        self.rprops = self._regionprops()
        self.features = self._features()

    def _normalize(self, image):
        image = resize(image, (self.dim,self.dim))
        image = self._to_binary(image)
        return image

    def _to_binary(self, image, thresh=0.5, invert=False):
        image = rgb2grey(image)
        if invert:
            return np.asarray(image < thresh, dtype='int')
        else:
            return np.asarray(image > thresh, dtype='int')

    @classmethod
    def from_file(self, filename, dim=64):
        image = imread(filename)
        # image = -image-1
        return NormalizedRegion(image, dim=dim)

    def _regionprops(self):
        regions = regionprops(self.image, properties=self.props)
        return regions[0]

    def _features(self):
        features = np.asarray(self.image.ravel(), dtype='float')
        extra_features = []
        for prop in self.props:
            extra_features.append(self.rprops[prop])
        return np.append(features, extra_features)

    def show(self):
        pyplot.imshow(self.image, cmap=cm.Greys, interpolation='nearest')

def save_regions(regions, prefix):
    for index, region in enumerate(regions):
        sp.misc.imsave('%s-%s.jpg' % (prefix, index), -region.image+1)

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
