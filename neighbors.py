#!/usr/bin/env python

"""
@package css.neighbors
@file css/neighbors.py
@author Edward Hunter
@author K Sree Harsha
@author Your Name Here
@brief Nearest neighbor and nearest centroid supervised learning
and evaluation methods.
"""

# Import common modules and utilities.
from common import *

# Define method and models available.
METHOD = 'Neighbors'
MODELS = ('KNN','Centroid')


def train(data, dataset, model, **kwargs):
    """
    Train and store feature extractor, dimension reducer and classifier.
    @param: data training and testing dataset dictionary.
    @param: dataset dataset name string, valid key to data.
    @param: model model name string.
    @param: fappend optional file name appendix string.
    @param: dim optional dimension integer for reduction.
    """

    # Verify input parameters.
    if not isinstance(data, dict):
        raise ValueError('Invalid data dictionary.')

    if not isinstance(dataset, str):
        raise ValueError('Invalid data dictionary.')

    if not isinstance(model,str) or model not in MODELS:
        raise ValueError('Invalid model type parameter.')

    # Retrieve training data.
    data_train = data['train']
    data_train_target = data['train_target']

    ############################################################
    # Create feature extractor, classifier.
    ############################################################
    vectorizer = TfidfVectorizer(stop_words='english',sublinear_tf=True)
    if model == 'KNN':
        clf = KNeighborsClassifier()
    elif model == 'Centroid':
        clf = NearestCentroid()
    ############################################################

    ############################################################
    # If specified, create feature dimension reducer.
    ############################################################
    dim = kwargs.get('dim', None)
    if dim:
        fselector = SelectKBest(chi2, k=dim)
    ############################################################

    ############################################################
    # Extract features, reducing dimension if specified.
    ############################################################
    print 'Extracting text features...'
    start = time.time()
    x_train = vectorizer.fit_transform(data_train)
    if dim:
        x_train = fselector.fit_transform(x_train, data_train_target)
        x_train = normalize(x_train)
    print 'Extracted in %f seconds.' % (time.time() - start)
    ############################################################

    ############################################################
    # Train classifier.
    ############################################################
    print 'Training classifier...'
    start = time.time()
    clf.fit(x_train, data_train_target)
    print 'Trained in %f seconds.' % (time.time() - start)
    ############################################################

    # Create classifier and feature extractor file names.
    fappend = kwargs.get('fappend', None)
    dim = kwargs.get('dim', None)
    (cfname, vfname, dfname, _) = get_fnames(METHOD, model, dataset, dim, fappend)

    if not os.path.exists(MODEL_HOME):
        os.makedirs(MODEL_HOME)

    # Write out classifier.
    cpname = os.path.join(MODEL_HOME,cfname)
    fhandle = open(cpname,'w')
    pickle.dump(clf, fhandle)
    fhandle.close()
    print 'Classifier written to file %s' % (cpname)

    # Write out feature extractor.
    vpname = os.path.join(MODEL_HOME,vfname)
    fhandle = open(vpname,'w')
    pickle.dump(vectorizer, fhandle)
    fhandle.close()
    print 'Feature extractor written to file %s' % (vpname)

    # Write out dimension reducer.
    dim = kwargs.get('dim', None)
    if dim:
        dpname = os.path.join(MODEL_HOME,dfname)
        fhandle = open(dpname,'w')
        pickle.dump(fselector, fhandle)
        fhandle.close()
        print 'Feature selector written to file %s' % (dpname)


def predict(input_data, cfname, vfname, **kwargs):
    """
    Predict data categories from trained classifier.
    @param: input_data vector of input data to classify.
    @param: cfname classifier filename.
    @param: vfname feature extractor filename.
    @param: dfname optional feature selector filename.
    @return: prediction vector for input_data.
    """
    # Verify input parameters.
    if not isinstance(input_data, list):
        raise ValueError('Invalid input data.')

    if not isinstance(cfname, str) or \
        not os.path.isfile(cfname):
        raise ValueError('Invalid classifier file name.')

    if not isinstance(vfname, str) or \
        not os.path.isfile(vfname):
        raise ValueError('Invalid feature extractor file name.')

    dfname = kwargs.get('dfname',None)
    if dfname:
        if not isinstance(dfname, str) or \
            not os.path.isfile(dfname):
            raise ValueError('Invalid dimension reducer file name.')

    # Read in the classifer.
    cpname = os.path.join(MODEL_HOME,cfname)
    fhandle = open(cpname)
    clf = pickle.load(fhandle)
    fhandle.close()
    print 'Read classifer from file: %s' % cpname

    # Read in the feature extractor.
    vpname = os.path.join(MODEL_HOME,vfname)
    fhandle = open(vpname)
    vectorizer = pickle.load(fhandle)
    fhandle.close()
    print 'Read feature extractor from file: %s' % vpname

    # If requested, load the dimension reducer.
    dfname = kwargs.get('dfname', None)
    if dfname:
        dpname = os.path.join(MODEL_HOME,dfname)
        fhandle = open(dpname, 'r')
        fselector = pickle.load(fhandle)
        fhandle.close()
        print 'Feature selector read from file %s' % (dpname)

    ############################################################
    # Compute features and predict.
    ############################################################
    x_test = vectorizer.transform(input_data)
    if dfname:
        x_test = fselector.transform(x_test)
        x_test = normalize(x_test)
    pred = clf.predict(x_test)
    ############################################################

    # Return vector of predicted labels.
    return pred


def eval(data, dataset, model, **kwargs):
    """
    Evaluate a trained classifer against test data.
    Prints out F1, precision, recall and confusion.
    Saves a png image of the confusion matrix.
    @param: data training and testing dataset dictionary.
    @param: dataset dataset name string, valid key to data.
    @param: model model name string.
    @param: fappend optional file name appendix string.
    @param: dim optional dimension integer for reduction.
    @param: confusion optional confusion image save boolean.
    """

    # Verify input parameters.
    if not isinstance(data, dict):
        raise ValueError('Invalid data dictionary.')

    if not isinstance(dataset, str):
        raise ValueError('Invalid data dictionary.')

    if not isinstance(model,str) or model not in MODELS:
        raise ValueError('Invalid model type parameter.')

    # Extract test and target data.
    data_test = data['test']
    data_test_target = data['test_target']
    data_target_names = data['target_names']

    # Create classifier, feature extractor and dim reducer names.
    fappend = kwargs.get('fappend', None)
    dim = kwargs.get('dim', None)
    (cfname, vfname, dfname, figfname) = get_fnames(METHOD, model, dataset, dim, fappend)

    # Predict test data.
    pred = predict(data_test, cfname, vfname, dfname=dfname)

    ############################################################
    # Evaluate predictions: metrics.
    ############################################################
    class_report = metrics.classification_report(data_test_target, pred,
                                                 target_names=data_target_names)
    conf_matrix = metrics.confusion_matrix(data_test_target ,pred)
    ############################################################

    # Print evaluations.
    print '-'*80
    print("Classification report:")
    print class_report

    print '-'*80
    print 'Confusion Matrix:'
    n = len(data_target_names)
    conf_max = np.amax(conf_matrix)
    lmax = math.log(conf_max, 10)
    width = int(lmax) + 1
    fmtstr ='%' + str(width) + 'd  '
    for j in range(n):
        row = ''
        for i in range(n):
            row += (fmtstr % int(conf_matrix[j,i]))
        print row

    # Save an image of the confusion matrix.
    if kwargs.get('confusion', False):
        plt.pcolor(np.flipud(conf_matrix))
        plt.xticks(np.arange(n)+0.5, np.arange(1,n+1))
        plt.yticks(np.arange(n)+0.5, np.arange(n,0, -1))
        plt.xlabel('Predicted Category')
        plt.ylabel('True Category')
        plt.set_cmap('hot')
        plt.colorbar()
        plt.title('%s %s Confusion, %s' % (METHOD, model, dataset))
        figpath = os.path.join(MODEL_HOME, figname)
        plt.savefig(figpath)


if __name__ == '__main__':

    # Load training/testing utilities.
    from data_utils import load_data, DATASETS

    # Parse command line arguments and options.
    usage = 'usage: %prog [options] model dataset'
    usage += ('\n\tmodel = %s\n\tdataset = %s') % (MODELS, DATASETS)
    description = 'Train and evaluate supervised classifiers.'
    p = optparse.OptionParser(usage=usage, description=description)
    p.add_option('-f','--fappend', action='store', dest='fappend',
                 help='File name appendix string.')
    p.add_option('-d','--dim', action='store', dest='dim', type='int',
                 help='Reduced feature dimension integer.')
    p.add_option('-c', '--confusion', action='store_true',
                 dest='confusion', help='Save confusion image.')
    p.add_option('-o', '--overwrite', action='store_true',
                 dest='overwrite', help='Overwrite existing files.')
    p.set_defaults(fappend=None, dim=None, confusion=True, overwrite=False)

    (opts, args) = p.parse_args()
    if len(args) < 2:
        p.print_usage()
        sys.exit(1)

    model = args[0]
    dataset = args[1]

    fappend = opts.fappend
    dim = opts.dim
    confusion = opts.confusion
    overwrite = opts.overwrite

    # Load data.
    data = load_data(dataset)

    # Create classifier, feature extractor and dim reducer names.
    (cfname, vfname, dfname, _) = get_fnames(METHOD, model, dataset, dim, fappend)

    # If we are specified to overwrite, or if required files missing, train and
    # store classifier components.
    if overwrite \
        or not(os.path.isfile(cfname) and os.path.isfile(vfname)) \
        or (dfname and not os.path.isfile(dfname)):
        train(data, dataset, model, dim=dim, fappend=fappend)

    # Evaluate classifier.
    eval(data, dataset, model, dim=dim, fappend=fappend, confusion=confusion)

