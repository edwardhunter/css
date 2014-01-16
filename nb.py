#!/usr/bin/env python

"""
@package css.nb
@file css/nb.py
@author Edward Hunter
@author K Sree Harsha
@brief Module and script for training Naive Bayes
classifiers for text data.
"""

__author__ = 'Edward Hunter'
__license__ = 'Apache 2.0'


# Import scikit modules for learning routines.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics

# Import numpy for vector manipulation.
import numpy as np

# Import matplotlib for plotting graphics.
import matplotlib.pyplot as plt

# Import python standard libraries for serialization, time, and system.
import pickle
import time
import sys
import os

MODELS = ('Bernoulli','Multinomial','TFIDF')

# Command line examples.
# python nb.py Bernoulli 20news
# python nb.py Multinomial 20news
# python nb.py TFIDF 20news
# python nb.py Bernoulli Reuters
# python nb.py Multinomial Reuters

def get_fnames(fname=None):
    """
    Return the file names of the stored classifier and feature extractor.
    @param fname the optional appendix to the classifier and features extractor file names.
    @retval triple containing classifier, extractor and figure filenames.
    """
    method = __file__.split('.')[0]
    if not fname:
        cfname = 'clf_%s_%s' % (method, model)
        vfname = 'vec_%s_%s' % (method, model)
        figfname = 'conf_%s_%s_%s.png' % (method, model, dataset)
    else:
        cfname = 'clf_%s_%s_%s' % (method, model, fname)
        vfname = 'vec_%s_%s_%s' % (method ,model, fname)
        figfname = 'conf_%s_%s_%s_%s.png' % (method, model, dataset, fname)

    return (cfname, vfname, figfname)


def train(data, dataset, model, **kwargs):
    """
    Train and store a naive bayes classifier with
    specified model and dataset.
    @param data training and testing dataset dictionary.
    @param dataset dataset name, valid key to data.
    @param model one of ('Bernoulli','Multinomial','TFIDF')
    @param fname optional file name appendix.
    Examples:
    train(data, '20news', 'bernoulli')
    train(data, '20news', 'bernoulli', '_test1')
    """

    # Verify input parameters.
    if not isinstance(data, dict):
        raise ValueError('Invalid data dictionary.')

    if not isinstance(dataset, str):
        raise ValueError('Invalid data dictionary.')

    if not dataset in data.keys():
        raise ValueError('Specified dataset not in data dictionary.')

    if not isinstance(model,str) or model not in MODELS:
        raise ValueError('Invalid model type parameter.')

    # Retrieve training data.
    data_train = data[dataset]['train']
    data_train_target = data[dataset]['train_target']

    ############################################################
    # Create feature extractor and classifier.
    ############################################################
    if model == 'Bernoulli':
        vectorizer = CountVectorizer(stop_words='english', binary=True)
        clf=BernoulliNB(alpha=.01)

    elif model == 'Multinomial':
        vectorizer = CountVectorizer(stop_words='english')
        clf=MultinomialNB(alpha=.01)

    elif model == 'TFIDF':
        vectorizer = TfidfVectorizer(stop_words='english')
        clf=MultinomialNB(alpha=.01)
    ############################################################

    ############################################################
    # Extract features.
    ############################################################
    print 'Extracting text features...'
    start = time.time()
    x_train = vectorizer.fit_transform(data_train)
    print 'Extracted in %f seconds.' % (time.time() - start)
    ############################################################

    ############################################################
    # Train classifier.
    ############################################################
    print 'Training %s Naive Bayes classifier...' % model
    start = time.time()
    clf.fit(x_train, data_train_target)
    print 'Trained in %f seconds.' % (time.time() - start)
    ############################################################

    # Create classifier and feature extractor file names.
    fname = kwargs.get('fname', None)
    (cfname, vfname, _) = get_fnames(fname)

    # Write out classifier.
    fhandle = open(cfname,'w')
    pickle.dump(clf, fhandle)
    fhandle.close()
    print '%s Naive Bayes classifier written to file %s' % (model, cfname)

    # Write out feature extractor.
    fhandle = open(vfname,'w')
    pickle.dump(vectorizer, fhandle)
    fhandle.close()
    print '%s Naive Bayes feature extractor written to file %s' % (model, vfname)

def predict(input_data, model, **kwargs):
    """
    Predict data categories from trained Naive Bayes classifier.
    @param input_data input feature vector to predict.
    @param model one of ('Bernoulli','Multinomial','TFIDF')
    @param fname optional file name appendix.
    @retval prediction vector for input_data.
    Examples:
    eval(data, '20news', 'bernoulli')
    eval(data, '20news', 'bernoulli', '_test1')
    """
    # Verify input parameters.
    if not isinstance(input_data, list):
        raise ValueError('Invalid input data.')

    if not isinstance(model,str) or model not in MODELS:
        raise ValueError('Invalid model type parameter.')

    fname = kwargs.get('fname', None)
    (cfname, vfname, _) = get_fnames(fname)

    # Read in the classifer.
    fhandle = open(cfname)
    clf = pickle.load(fhandle)
    fhandle.close()
    print 'Read classifer from file: %s' % cfname

    # Read in the feature extractor.
    fhandle = open(vfname)
    vectorizer = pickle.load(fhandle)
    fhandle.close()
    print 'Read feature extractor from file: %s' % vfname

    ############################################################
    # Compute features and predict.
    ############################################################
    x_test = vectorizer.transform(input_data)
    pred = clf.predict(x_test)
    ############################################################

    return pred


def eval(data, dataset, model, **kwargs):
    """
    Evaluate a trained classifer against test data.
    Prints out F1, precision, recall and confusion.
    Saves a png image of the confusion matrix.
    @param data training and testing dataset dictionary.
    @param dataset dataset name, valid key to data.
    @param model one of ('Bernoulli','Multinomial','TFIDF')
    @param fname optional file name appendix.
    Examples:
    eval(data, '20news', 'Bernoulli')
    eval(data, '20news', 'Bernoulli', '_test1')
    """

    # Verify input parameters.
    if not isinstance(data, dict):
        raise ValueError('Invalid data dictionary.')

    if not isinstance(dataset, str):
        raise ValueError('Invalid data dictionary.')

    if not dataset in data.keys():
        raise ValueError('Specified dataset not in data dictionary.')

    if not isinstance(model,str) or model not in MODELS:
        raise ValueError('Invalid model type parameter.')

    # Extract test and target data.
    data_test = data[dataset]['test']
    data_test_target = data[dataset]['test_target']
    data_target_names = data[dataset]['target_names']

    # Predict test data.
    fname = kwargs.get('fname', None)
    pred = predict(data_test, model, fname=fname)

    ############################################################
    # Evaluate predictions: metrics.
    ############################################################
    f1 = metrics.f1_score(data_test_target, pred)
    precision = metrics.precision_score(data_test_target, pred)
    recall = metrics.recall_score(data_test_target, pred)
    ############################################################

    ############################################################
    # Evaluate predictions: confusion.
    ############################################################
    n = len(data_target_names)
    conf_matrix = np.zeros((n,n),dtype=np.int32)
    for i in range(len(pred)):
        true_val = data_test_target[i]
        pred_val = pred[i]
        conf_matrix[true_val, pred_val] += 1
    ############################################################

    # Print evaluation data.
    print '-'*80
    print("F1-score:  \t\t %0.3f" % f1)
    print("Precision: \t\t %0.3f " % precision)
    print("Recall:    \t\t %0.3f " % recall)
    print ''

    print '-'*80
    print 'Confusion Matrix:'
    np.set_printoptions(linewidth=150)
    print conf_matrix

    # Save an image of the confusion matrix.
    plt.pcolor(np.flipud(conf_matrix))
    plt.xticks(np.arange(n)+0.5, np.arange(1,n+1))
    plt.yticks(np.arange(n)+0.5, np.arange(n,0, -1))
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')
    plt.set_cmap('hot')
    plt.colorbar()
    plt.title('%s Naive Bayes Confusion, %s' % (model, dataset))

    (_, _, figfname) = get_fnames(fname)

    plt.savefig(figfname)


def usage():
    """
    Print usage message.
    """
    print 'Usage:'
    print 'python nb.py model dataset [fileid]'
    print 'model: one of ' + str(MODELS)
    print 'dataset: a named dataset.'
    print 'fileid: an optional appendix identifying classifier and feature extractor files.'
    print 'default file names are clf_nb_model_dataset and vec_nb_model_dataset'

if __name__ == '__main__':

    # Print usage if required arguments missing.
    if len(sys.argv) < 3:
        usage()
        sys.exit(-1)

    model = sys.argv[1]
    dataset = sys.argv[2]

    # Get file name appendix.
    fname = None
    if len(sys.argv) >= 4:
        fname = sys.argv[3]

    # Create classifier and feature extractor names.
    (cfname, vfname, _) = get_fnames(fname)

    # Import the data.
    from data import data

    # If classifier not trained and saved, create it.
    if not (os.path.isfile(cfname) and os.path.isfile(vfname)):
        train(data, dataset, model, fname=fname)

    # Evaluate classifier.
    eval(data, dataset, model, fname=fname)

