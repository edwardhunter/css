#!/usr/bin/env python

"""
@package css
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
from sklearn.grid_search import GridSearchCV

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
from collections import Counter

# Default model directory.
MODEL_HOME = os.path.join('.', 'models')
REPORT_HOME =  os.path.join('.', 'reports')

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

    base_name = '%s_%s_%s' % (method, model, dataset)

    if dim:
        base_name = '%s_%i' % (base_name, dim)

    if appendix:
        base_name = '%s_%s' % (base_name, appendix)

    cfname = '%s_clf' % (base_name)
    vfname = '%s_vec' % (base_name)
    dfname = '%s_dim' % (base_name)
    figfname = '%s_confusion.png' % (base_name)
    reportfname = '%s_report.txt' % (base_name)
    if not dim:
        dfname = None

    return (cfname, vfname, dfname, figfname, reportfname)
