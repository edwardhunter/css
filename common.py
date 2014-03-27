#!/usr/bin/env python

"""
@package css
@file css/common.py
@author Edward Hunter
@brief Common imports and utility functions.
"""

# Copyright and licence.
"""
Copyright (C) 2014 Edward Hunter
edward.a.hunter@gmail.com
840 24th Street
San Diego, CA 92102

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

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
import urllib
from collections import Counter

# Import numpy for vector manipulation.
import numpy as np

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
from sklearn.utils.extmath import density

# Spherical k means support.
from spkm import spkmeans

# Import matplotlib for plotting graphics.
import matplotlib.pyplot as plt

# Import BeautifulSoup for HTML parsing.
from bs4 import BeautifulSoup


# Default model directory.
MODEL_HOME = os.path.join('.', 'models')
REPORT_HOME =  os.path.join('.', 'reports')

def get_fnames(method, model, dataset, dim=None, appendix=None):
    """
    Return the file names of the stored classifier, feature extractor,
    dimesnion reducer, confusion image and report.
    @param: method the learning method, usually the module file name.
    @param: model the learning model.
    @param: dataset the dataset used.
    @param: dim the reduce dimension to specified integer.
    @param: appendix the optional appendix to the classifier and features extractor file names.
    @return: 5-tuple containing file names.
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

def make_fname(method, model, dataset, suffix, ext=None, *args):
    """
    Return filename for objects to be stored.
    @param method: the learning method, usually the module file name.
    @param model: the learning model.
    @param dataset: the dataset used.
    @param suffix: the type of object.
    @param ext: the filename extension.
    @return fname: the filename.
    """

    fname = '%s_%s_%s' % (method, model, dataset)
    args = [x for x in args if x]
    for x in args:
        fname += '_' + x

    fname += '_' + suffix
    if ext:
        fname += '.' + ext
    return fname

