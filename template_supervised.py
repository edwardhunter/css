#!/usr/bin/env python

"""
@package css
@file css/template_supervised.py
@author Edward Hunter
@brief A template to be customized for supervised learning experiments.
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

# Import common modules and utilities.
from common import *

# Define method and models available.
METHOD = ''
MODELS = ()


def train(data, dataset, model, **kwargs):
    """
    Train and store feature extractor, dimension reducer and classifier.
    @param data: training and testing dataset dictionary.
    @param dataset: dataset name string, valid key to data.
    @param model: model name string.
    """

    # Verify input parameters.
    if not isinstance(data, dict):
        raise ValueError('Invalid data dictionary.')

    if not isinstance(dataset, str) or dataset not in DATASETS:
        raise ValueError('Invalid dataset.')

    if not isinstance(model,str) or model not in MODELS:
        raise ValueError('Invalid model type parameter.')

    # Retrieve training data.
    data_train = data['train']
    data_train_target = data['train_target']

    # Retrieve options.
    dim = kwargs.get('dim', None)
    df_min = kwargs.get('df_min',1)
    df_max = kwargs.get('df_max',1.0)

    # Create dimension reducer.
    fselector = None
    if dim:
        fselector = SelectKBest(chi2, k=dim)

    ############################################################
    # Create feature extractor, classifier.
    ############################################################
    # TODO: create clf, vectorizer.
    ############################################################

    # Extract features, reducing dimension if specified.
    print 'Extracting text features...'
    start = time.time()
    x_train = vectorizer.fit_transform(data_train)
    if fselector:
        x_train = fselector.fit_transform(x_train, data_train_target)
        if fselector.__normalize:
            x_train = normalize(x_train)
    print 'Extracted in %f seconds.' % (time.time() - start)
    print 'Feature dimension: %i' %x_train.shape[1]
    print 'Feature density: %f' % density(x_train)

    # Train classifier.
    print 'Training classifier...'
    start = time.time()
    clf.fit(x_train, data_train_target)
    print 'Trained in %f seconds.' % (time.time() - start)

    # Default grid search and top features output triggers.
    grid_search_output = False
    top_features_output = False

    ############################################################
    # Grid search and top feature output for SVMs.
    ############################################################
    # TODO: SVMs only.
    ############################################################

    # Print out grid search results.
    if grid_search_output:
        print 'Best score: ' + str(clf.best_score_)
        print 'Optimal parameters: '
        for k,v in clf.best_params_.iteritems():
            print '%s=%s' % (k, str(v))

    # Print out top features results.
    if top_features_output:
        print("Classifier shape: %s" % str(clf.coef_.shape))
        feature_names = np.asarray(vectorizer.get_feature_names())
        top = clf.coef_.toarray().argsort(axis=1)[0]
        top_pos = top[-svm_top:]
        top_neg = top[:svm_top]
        print '-'*40
        print 'Top %s Features:' % data['target_names'][1]
        for idx in top_pos:
            print feature_names[idx]
        print '-'*40
        print 'Top %s Features:' % data['target_names'][0]
        for idx in top_neg:
            print feature_names[idx]

    # Create object file names.
    fname_args = []
    if dim:
        fname_args.append(str(dim))
    fname_args.append(kwargs.get('fappend', ''))
    mdl_fname = make_fname(METHOD, model, dataset, 'mdl', 'pk', *fname_args)
    vec_fname = make_fname(METHOD, model, dataset, 'vec', 'pk', *fname_args)
    dim_fname = make_fname(METHOD, model, dataset, 'dim', 'pk', *fname_args)
    mdl_path = os.path.join(MODEL_HOME,mdl_fname)
    vec_path = os.path.join(MODEL_HOME,vec_fname)
    dim_path = os.path.join(MODEL_HOME,dim_fname)

    if not os.path.exists(MODEL_HOME):
        os.makedirs(MODEL_HOME)

    # Write out model.
    fhandle = open(mdl_path,'w')
    pickle.dump(clf, fhandle)
    fhandle.close()
    print 'Model written to file %s' % (mdl_path)

    # Write out feature extractor.
    fhandle = open(vec_path,'w')
    pickle.dump(vectorizer, fhandle)
    fhandle.close()
    print 'Feature extractor written to file %s' % (vec_path)

    # Write out dimension reducer.
    if dim:
        fhandle = open(dim_path,'w')
        pickle.dump(fselector, fhandle)
        fhandle.close()
        print 'Dimension reducer written to file %s' % (dim_path)


def predict(input_data, dataset, model, **kwargs):
    """
    Predict data categories from trained classifier.
    @param input_data: vector of input data to classify.
    @param dataset: dataset name string, valid key to data.
    @param model: model name string.
    @return pred: prediction vector for input_data.
    """

    # Verify input parameters.
    if not isinstance(input_data, list):
        raise ValueError('Invalid input data.')

    if not isinstance(dataset, str) or dataset not in DATASETS:
        raise ValueError('Invalid dataset name.')

    if not isinstance(model,str) or model not in MODELS:
        raise ValueError('Invalid model name.')

    # Retrieve options.
    dim = kwargs.get('dim', None)
    fappend = kwargs.get('fappend', '')

    # Create object file names.
    fname_args = []
    if dim:
        fname_args.append(str(dim))
    fname_args.append(fappend)
    mdl_fname = make_fname(METHOD, model, dataset, 'mdl', 'pk', *fname_args)
    vec_fname = make_fname(METHOD, model, dataset, 'vec', 'pk', *fname_args)
    dim_fname = make_fname(METHOD, model, dataset, 'dim', 'pk', *fname_args)
    mdl_path = os.path.join(MODEL_HOME,mdl_fname)
    vec_path = os.path.join(MODEL_HOME,vec_fname)
    dim_path = os.path.join(MODEL_HOME,dim_fname)

    # Read in the classifer.
    fhandle = open(mdl_path)
    clf = pickle.load(fhandle)
    fhandle.close()
    print 'Read classifer from file: %s' % mdl_path

    # Read in the feature extractor.
    fhandle = open(vec_path)
    vectorizer = pickle.load(fhandle)
    fhandle.close()
    print 'Read feature extractor from file: %s' % vec_path

    # If requested, load the dimension reducer.
    if dim:
        fhandle = open(dim_path, 'r')
        fselector = pickle.load(fhandle)
        fhandle.close()
        print 'Read dimesion reducer from file %s' % (dim_path)

    # Compute features and predict.
    x_test = vectorizer.transform(input_data)
    if dim:
        x_test = fselector.transform(x_test)
        if fselector.__normalize:
            x_test = normalize(x_test)
    pred = clf.predict(x_test)

    # Return vector of predicted labels.
    return pred


def eval(data, dataset, model, **kwargs):
    """
    Evaluate a trained classifer against test data.
    Prints out F1, precision, recall and confusion.
    Saves a png image of the confusion matrix.
    @param: data training and testing dataset dictionary.
    @param dataset: dataset name string, valid key to data.
    @param model: model name string.
    """

    # Verify input parameters.
    if not isinstance(data, dict):
        raise ValueError('Invalid data dictionary.')

    if not isinstance(dataset, str) or dataset not in DATASETS:
        raise ValueError('Invalid dataset name.')

    if not isinstance(model,str) or model not in MODELS:
        raise ValueError('Invalid model name.')

    # Extract test and target data.
    data_test = data['test']
    data_test_target = data['test_target']
    data_target_names = data['target_names']

    # Predict test data.
    pred = predict(data_test, dataset, model, **kwargs)

    # Evaluate predictions: metrics.
    class_report = metrics.classification_report(data_test_target, pred,
                                                 target_names=data_target_names)
    conf_matrix = metrics.confusion_matrix(data_test_target ,pred)

    # Set up report and figure name, path.
    fname_args = []
    if dim:
        fname_args.append(str(dim))
    fname_args.append(fappend)
    reportname = make_fname(METHOD, model, dataset, 'report', 'txt', *fname_args)
    reportpath = os.path.join(REPORT_HOME, reportname)
    figfname = make_fname(METHOD, model, dataset, 'confusion', 'png', *fname_args)
    figpath = os.path.join(REPORT_HOME, figfname)

    # Print evaluations.
    report = 'Report File: %s\n' % reportname
    report += '-'*80 +'\n'
    report += 'Classification Report:\n'
    report += class_report
    report += '\n\n'
    report += '-'*80 +'\n'
    report += 'Confusion Matrix:\n'
    n = len(data_target_names)
    conf_max = np.amax(conf_matrix)
    lmax = math.log(conf_max, 10)
    width = int(lmax) + 1
    fmtstr ='%' + str(width) + 'd  '
    for j in range(n):
        row = ''
        for i in range(n):
            row += (fmtstr % int(conf_matrix[j,i]))
        report += row + '\n'

    report += '\n\n'
    print report

    if not os.path.exists(REPORT_HOME):
        os.makedirs(REPORT_HOME)

    # Write report.
    rf = open(reportpath, 'w')
    rf.write(report)
    rf.close()

    # Create an image of the log confusion matrix.
    confusion_image_type =  kwargs.get('confusion', None)
    if not confusion_image_type:
        pass
    elif confusion_image_type not in ('linear','log'):
        warnstr = 'WARNING: unrecognized confusion image option '
        warnstr += '"%s"' % confusion_image_type
        warnstr += '\nConfusion image not saved.'
        print warnstr
    else:
        if confusion_image_type == 'log':
            log_conf_matrix = np.log10(conf_matrix+1)
            plt.pcolor(np.flipud(log_conf_matrix))
            title = '%s %s Log Confusion, %s' % (METHOD, model, dataset)
        elif confusion_image_type == 'linear':
            plt.pcolor(np.flipud(conf_matrix))
            title = '%s %s Confusion, %s' % (METHOD, model, dataset)
        plt.xticks(np.arange(n)+0.5, np.arange(1,n+1))
        plt.yticks(np.arange(n)+0.5, np.arange(n,0, -1))
        plt.xlabel('Predicted Category')
        plt.ylabel('True Category')
        plt.set_cmap('hot')
        plt.colorbar()
        plt.title(title)
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
    p.add_option('-c', '--confusion', action='store', dest='confusion',
                 help='Save confusion image. Options: linear, log')
    p.add_option('-o', '--overwrite', action='store_true', dest='overwrite',
                 help='Overwrite existing files.')
    p.add_option('--df_min', action='store',type='float', dest='df_min',
                 help='Minimum frequency (int) or proportion (float) (default=1.0).')
    p.add_option('--df_max', action='store', type='float', dest='df_max',
                 help='Maximum document frequency proportion (default=1.0).')
    p.set_defaults(fappend=None, dim=None, confusion=None, overwrite=False,
                   df_min=1.0, df_max=1.0)

    ############################################################
    # Add method specific options.
    ############################################################
    # TODO: SVMs only.
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
    if opts.df_min == int(opts.df_min):
        df_min = int(opts.df_min)
    else:
        df_min = opts.df_min
    df_max = opts.df_max
    kwargs = dict(
        dim=dim,
        fappend=fappend,
        df_min=df_min,
        df_max=df_max,
        confusion=confusion,
    )

    ############################################################
    # Extract method specific options.
    ############################################################
    # TODO: SVMs only.
    ############################################################

    # Load data.
    data = load_data(dataset)

    # Create object file names.
    fname_args = []
    if dim:
        fname_args.append(str(dim))
    fname_args.append(fappend)
    mdl_fname = make_fname(METHOD, model, dataset, 'mdl', 'pk', *fname_args)
    vec_fname = make_fname(METHOD, model, dataset, 'vec', 'pk', *fname_args)
    dim_fname = make_fname(METHOD, model, dataset, 'dim', 'pk', *fname_args)
    mdl_path = os.path.join(MODEL_HOME,mdl_fname)
    vec_path = os.path.join(MODEL_HOME,vec_fname)
    dim_path = os.path.join(MODEL_HOME,dim_fname)

    model_files_present = os.path.isfile(mdl_path) and os.path.isfile(vec_path)
    if dim:
        dim_files_present = model_files_present and os.path.isfile(dim_path)

    if overwrite or not model_files_present:
        train(data, dataset, model, **kwargs)

    # Evaluate classifier.
    eval(data, dataset, model, **kwargs)

