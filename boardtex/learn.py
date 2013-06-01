# coding: utf-8

from __future__ import division, print_function

# Standard library
import os, sys
import glob

# Third-party
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.svm import SVC

from symbols import symbol_to_idx

# Function that accepts an image and a list of symbols and
#   returns a list of Regions and an array of labels

'''
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
    data = np.array([NormalizedRegion(regions[i]).features for i in sort_idx])
    
    if len(data) != len(labels):
        raise ValueError("Found {0} regions in file {1}, but expected "
                         "{2} symbols.".format(len(data),filename,len(labels)))
    
    return data, labels

'''

def train(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    
    tuned_parameters = [{'kernel': ['rbf'], 
                         'gamma': np.logspace(-5,0,10),
                         'C': np.logspace(0,3,10)}]
    
    clf = GridSearchCV(SVC(C=1), tuned_parameters, score_func=precision_score)
    clf.fit(X_train, y_train, cv=5)
    
    # TODO: turn these into logging...
    #print("Best parameter set: \n{0}".format(clf.best_estimator_))
    #for params, mean_score, scores in clf.grid_scores_:
    #    print("{0:.3f} (+/-{1:.3f}) for {2}".format(mean_score, scores.std() / 2, params))
    
    return clf.best_estimator_

def test_train():
    from sklearn import datasets
    from skimage.data import imread
    from skimage.color import rgb2grey
    
    # Loading the Digits dataset
    digits = datasets.load_digits()
    
    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target
    
    clf = train(X,y)
    
    # Read one of my digits, predict on that
    one_data = rgb2grey(imread("/Users/adrian/Downloads/digits/one.png"))
    print(clf.predict(one_data.ravel()))
    