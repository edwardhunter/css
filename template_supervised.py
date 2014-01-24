#!/usr/bin/env python

"""
@package css.template_supervised
@file css/template_supervised.py
@author Edward Hunter
@author Your Name Here
@brief A template to be customized for supervised learning experiments.
"""

# Import common modules and utilities.
from common import *

# Define method and models available.
METHOD = ''
MODELS = ()


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
    # TODO: create clf, vectorizer.
    ############################################################

    ############################################################
    # If specified, create feature dimension reducer.
    ############################################################
    # TODO: create fselector.
    ############################################################

    ############################################################
    # Extract features, reducing dimension if specified.
    ############################################################
    # TODO
    ############################################################

    ############################################################
    # Train classifier.
    ############################################################
    # TODO
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

    if not isinstance(cfname, str):
        raise ValueError('Invalid classifier file name.')

    if not isinstance(vfname, str):
        raise ValueError('Invalid feature extractor file name.')

    dfname = kwargs.get('dfname',None)
    if dfname and not isinstance(dfname, str):
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
    # TODO: create variable pred.
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
        raise ValueError('Invalid dataset name.')

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
    # TODO: create values for f1, precision, recall, conf_matrix
    # TODO: Create variables class_report, conf_matrix.
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
        figpath = os.path.join(MODEL_HOME, figfname)
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

    ############################################################
    # Add method specific options.
    ############################################################
    # TODO
    ############################################################


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

    ############################################################
    # Extract method specific options.
    ############################################################
    # TODO (optional): create options dict method_kwargs for train.
    ############################################################

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

