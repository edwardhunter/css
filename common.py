#!/usr/bin/env python

"""
@package css.common
@file css/common.py
@author Edward Hunter
@author K Sree Harsha
@brief Common imports and utility functions.
"""

# Import scikit modules for learning routines.
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import download_20newsgroups
from twenty_news5groups import *
from twenty_news4groups import *


# Import numpy for vector manipulation.
import numpy as np

# Import matplotlib for plotting graphics.
import matplotlib.pyplot as plt

# Import python standard libraries.
import pickle
import time
import sys
import os
import optparse
import math
import tarfile
import re
import logging
import copy
import random
import shutil
from urllib2 import urlopen

# Default model directory.
MODEL_HOME = os.path.join('.', 'models')

def get_fnames(method, model, dataset, dim=None, appendix=None):
    """
    Return the file names of the stored classifier and feature extractor.
    @param: method the learning method, usually the module file name.
    @param: model the learning model.
    @param: dataset the dataset used.
    @param: dim the reduce dimension to specified integer.
    @param: appendix the optional appendix to the classifier and features extractor file names.
    @return: triple containing classifier, extractor and figure filenames.
    """

    cfname = '%s_%s_%s' % (method, model, dataset)
    vfname = '%s_%s_%s' % (method, model, dataset)
    dfname = '%s_%s_%s' % (method, model, dataset)
    figfname = '%s_%s_%s' % (method, model, dataset)

    if dim:
        cfname = '%s_%i' % (cfname, dim)
        vfname = '%s_%i' % (vfname, dim)
        dfname = '%s_%i' % (dfname, dim)
        figfname = '%s_%i' % (figfname, dim)

    if appendix:
        cfname = '%s_%s' % (cfname, appendix)
        vfname = '%s_%s' % (vfname, appendix)
        dfname = '%s_%s' % (dfname, appendix)
        figfname = '%s_%s' % (figfname, appendix)

    cfname = '%s_clf' % (cfname)
    vfname = '%s_vec' % (vfname)
    dfname = '%s_dim' % (dfname)
    figfname = '%s_con.png' % (figfname)

    if not dim:
        dfname = None

    return (cfname, vfname, dfname, figfname)
