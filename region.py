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
        self.rprops = self._regionprops(image)

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
    def from_file(self, filename):
        pass

    def _regionprops(self, image):
        regions = regionprops(image, properties=self.props)
        return regions[0]

    @property
    def features(self):
        image_size = self.dim*self.dim
        size = image_size + len(self.props)
        features = np.zeros(size, dtype='float')
        features[:image_size] = self.image
        for i, prop in enumerate(self.props):
            features[image_size+i] = self.rprops[prop]
        return features

    def show(self):
        pyplot.imshow(self.image, cmap=cm.Greys)

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
