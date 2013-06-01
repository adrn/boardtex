# coding: utf-8

from __future__ import division, print_function

# Standard library
import os, sys
import glob

# Third-party
import numpy as np

from symbols import symbol_to_index

# Function that accepts an image and a list of symbols and
#   returns a list of Regions and an array of labels

def image_to_training_data(filename, symbols):
    
    # Turn the list of expected symbols into an array of class labels
    labels = np.array([symbol_to_index[s] for s in symbols])
    
    # generate a list of Regions from the filename
    image_data = imread(filename)
    regions = ...split(image_data)
    
    # order the regions by index increasing across row, and down columns
    coordinates = [(r._,r._) for r in regions]
    sort_idx = [i[0] for i in sorted(enumerate(coordinates), key=lambda x:x[1])]
    
    # extract the features in order
    data = np.array([regions[i].features for i in sort_idx])
    
    if len(data) != len(labels):
        raise ValueError("Found {0} regions in file {1}, but expected "
                         "{2} symbols.".format(len(data),filename,len(labels)))
    
    return data, labels
    
""" API:

X,y = image_to_training_data('greek_symbols.png', ['alpha','beta',\
                                                   'gamma','delta',\
                                                   'epsilon','sqrt{x}',\
                                                   '\frac{1}{2}'])
"""
                                            