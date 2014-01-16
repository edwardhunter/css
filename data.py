#!/usr/bin/env python

"""
@package css.data
@file css/data.py
@author Edward Hunter
@author K Sree Harsha
@brief Module for retrieving and loading training and testing
text datasets.
"""

# Import scikit modules.
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm
from math import sqrt
from sklearn.preprocessing import normalize
MODELS = ('Bernoulli','Multinomial','TFIDF')


def show_features(data, dataset, model, idx=[0]):
    """
    Print out example computed features for specified training data.
    This allows us to visualize that the features are
    computed correctly.
    @param data training and testing dataset dictionary.
    @param dataset dataset name, valid key to data.
    @param model one of ('Bernoulli','Multinomial','TFIDF')
    @param idx optional list of training data features to print. Default [0].
    Examples:
    show_features(data, '20news', 'Bernoulli', [0, 20, 40])
    show_features(data, '20news', 'Multinomial', [0, 20, 40])
    show_features(data, 'reuters', 'Bernoulli', [0, 20, 40])
    show_features(data, 'reuters', 'Multinomial', [0, 20, 40])
    """

    # Verify input parameters.
    if not data:
        raise ValueError('Invalid data dictionary.')

    if not dataset:
        raise ValueError('Invalid data dictionary.')

    if not dataset in data.keys():
        raise ValueError('Specified dataset not in data dictionary.')

    if not model or model not in MODELS:
        raise ValueError('Invalid model type parameter.')

    if not isinstance(idx,(list, tuple)):
        raise ValueError('Requries list of example data indices.')

    # Retrieve training data.
    data_train = data[dataset]['train']

    ############################################################
    # Create feature extractor.
    ############################################################
    if model == 'Bernoulli':
        vectorizer = CountVectorizer(stop_words='english', binary=True)

    elif model == 'Multinomial':
        vectorizer = CountVectorizer(stop_words='english')

    elif model == 'TFIDF':
        vectorizer = TfidfVectorizer(stop_words='english')

    x_train = vectorizer.fit_transform(data_train)
    ############################################################

    # Print feature dimension,
    print '-'*80
    print dataset + ' feature dimension size: %i' % x_train.shape[1]

    # Print example text and features.
    for i in idx:
        print '-'*80
        print 'Example data %i' % i
        print data_train[i]
        print '-'*80
        print model + ' feature:'
        print x_train[i,:]
        xx = x_train[i, :].copy()
        xn = normalize(xx, norm='l2')
        print xn
        #print str(np.linalg.norm(x_train[i,:]))
    print '-'*80


def load_data():
    """
    Load dictionary of all training and test data.
    """

    # Create an empty dictionary to hold all the data.
    data = {}

    # Fetch the 20 newsgroup data, removing headers, footers, quotes.
    categories = ['comp.os.ms-windows.misc',
                  'rec.sport.hockey'
                  ]
    categories = None
    remove = ('headers', 'footers', 'quotes')
    #remove = ()
    news_data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

    news_data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

    # Populate the 20 newsgroup data into our result data dictionary.
    data['20news'] = {}
    data['20news']['target_names'] = news_data_train.target_names
    data['20news']['train'] = news_data_train.data
    data['20news']['test'] = news_data_test.data
    data['20news']['train_target'] = news_data_train.target
    data['20news']['test_target'] = news_data_test.target


    # Print out data sizes.
    for key,val in data.iteritems():
        print '-'*80
        print key + ' target categories:'
        for target in data[key]['target_names']:
            print target
        print '-'*80
        print key + ' training data size: ' + str(len(data[key]['train']))
        print key + ' test data size: ' + str(len(data[key]['test']))
        print '-'*80


    # Return the data dictionary.
    return data

data = load_data()


"""
vec = TfidfVectorizer(stop_words='english')

train_data = data['20news']['train']
train_target = data['20news']['train_target']

x_train = vec.fit_transform(train_data)

print str(x_train.shape)
xx = x_train[0,:].copy()
yy = xx.copy()
print str(xx.shape)
print str(yy.shape)
print type(xx)
zz = xx.dot(yy.transpose())
print str(zz.shape)
print str(zz[0,0])
print xx
#n = sqrt(xx.transpose(copy=True).dot(xx))
#print str(n)

nn = 0.0
(r,c) = xx.shape
print str(r)
print str(c)
for i in range(c):
    nn += xx[0,i]**2
print str(nn)
"""