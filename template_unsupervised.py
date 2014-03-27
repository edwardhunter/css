#!/usr/bin/env python

"""
@package css
@file css/template_unsupervised.py
@author Edward Hunter
@brief A template to be customized for unsupervised learning experiments.
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


def train(data, dataset, model, no_components, no_runs, **kwargs):
    """
    Train and store feature extractor, dimension reducer and unsupervised
    model.
    @param data: training and testing dataset dictionary.
    @param dataset: dataset name string, valid key to data.
    @param model: model name string.
    @param no_components: number of model components.
    @param no_runs: number of training runs.
    """

    # Verify input parameters.
    if not isinstance(data, dict):
        raise ValueError('Invalid data dictionary.')

    if not isinstance(dataset, str) or dataset not in DATASETS:
        raise ValueError('Invalid dataset.')

    if not isinstance(model,str) or model not in MODELS:
        raise ValueError('Invalid model type parameter.')

    # Retrieve training data.
    _data = data['data']

    # Retrieve options.
    fappend = kwargs.get('fappend', '')
    df_min = kwargs.get('df_min',1)
    df_max = kwargs.get('df_max',1.0)

    ############################################################
    # Create feature extractor, learner.
    ############################################################
    # TODO: create vectorizer, unsupervised learner.
    ############################################################

    # Extract features, reducing dimension if specified.
    print 'Extracting text features...'
    start = time.time()
    x = vectorizer.fit_transform(_data)
    print 'Extracted in %f seconds.' % (time.time() - start)
    print 'Feature dimension: %i' %x.shape[1]
    print 'Feature density: %f' % density(x)

    # Define model output container.
    results =dict(
        models=[],
        labels=[],
        scores=[],
        sizes=[]
    )

    # Loop model.
    print 'Learning %s model %s, %i runs...' % (METHOD, model, no_runs)
    for i in range(no_runs):
        starttime = time.time()
        ############################################################
        # Run algorithm.
        ############################################################
        # TODO: Call unsupervised algorithm.
        ############################################################
        results['models'].append(model_i)
        results['labels'].append(labels_i)
        results['scores'].append(scores_i)
        results['sizes'].append(sizes_i)

        print 'Run %i/%i in %f seconds.' % (i+1, no_runs, time.time() - starttime)
    ############################################################

    # Create object file names.
    fname_args = []
    fname_args.append(str(no_components))
    fname_args.append(fappend)
    mdl_fname = make_fname(METHOD, model, dataset, 'mdl', 'pk', *fname_args)
    vec_fname = make_fname(METHOD, model, dataset, 'vec', 'pk', *fname_args)
    mdl_path = os.path.join(MODEL_HOME,mdl_fname)
    vec_path = os.path.join(MODEL_HOME,vec_fname)

    if not os.path.exists(MODEL_HOME):
        os.makedirs(MODEL_HOME)

    # Write out model.
    fhandle = open(mdl_path,'w')
    pickle.dump(model, fhandle)
    fhandle.close()
    print 'Model written to file %s' % (mdl_path)

    # Write out feature extractor.
    fhandle = open(vec_path,'w')
    pickle.dump(vectorizer, fhandle)
    fhandle.close()
    print 'Feature extractor written to file %s' % (vec_path)


def eval(dataset, model, no_components, **kwargs):
    """
    Evaluate unsupervised model.
    @param dataset: The dataset name.
    @param model: The model name.
    @param no_components: Number of model components.
    """

    # Retrieve options.
    fappend = kwargs.get('fappend', '')

    # Create object file names.
    fname_args = []
    fname_args.append(str(no_components))
    fname_args.append(fappend)
    mdl_fname = make_fname(METHOD, model, dataset, 'mdl', 'pk', *fname_args)
    vec_fname = make_fname(METHOD, model, dataset, 'vec', 'pk', *fname_args)
    mdl_path = os.path.join(MODEL_HOME,mdl_fname)
    vec_path = os.path.join(MODEL_HOME,vec_fname)

    # Read in the results.
    fhandle = open(mdl_path)
    results = pickle.load(fhandle)
    fhandle.close()
    print 'Read model from file: %s' % mdl_path

    # Read in the feature extractor.
    fhandle = open(vec_path)
    vectorizer = pickle.load(fhandle)
    fhandle.close()
    print 'Read feature extractor from file: %s' % vec_path

    ############################################################
    # Create report data.
    ############################################################
    # TODO: create reports and plots.
    ############################################################


if __name__ == '__main__':

    # Load training/testing utilities.
    from data_utils import load_unsupervised_data, DATASETS

    # Parse command line arguments and options.
    usage = 'usage: %prog [options] no_components no_runs dataset'
    usage += '\n\tno_components = int number of model components.'
    usage += '\n\tno_runs = int number of model runs.'
    usage += '\n\tdataset = %s.' % str(DATASETS)
    description = 'Train and evaluate supervised classifiers.'
    p = optparse.OptionParser(usage=usage, description=description)
    p.add_option('-f','--fappend', action='store', dest='fappend',
                 help='File name appendix string.')
    p.add_option('-o', '--overwrite', action='store_true', dest='overwrite',
                 help='Overwrite existing files.')
    p.add_option('--df_min', action='store',type='float', dest='df_min',
                 help='Minimum frequency (int) or proportion (float) (default=1.0).')
    p.add_option('--df_max', action='store', type='float', dest='df_max',
                 help='Maximum document frequency proportion (default=1.0).')
    p.set_defaults(fappend='', overwrite=False, df_min=1.0, df_max=1.0)

    (opts, args) = p.parse_args()
    if len(args) < 4:
        p.print_usage()
        sys.exit(1)

    model = args[0]
    dataset = args[1]
    no_components = args[2]
    no_runs = args[3]

    fappend = opts.fappend
    overwrite = opts.overwrite
    if opts.df_min == int(opts.df_min):
        df_min = int(opts.df_min)
    else:
        df_min = opts.df_min
    df_max = opts.df_max
    kwargs = {}
    kwargs['fappend'] = fappend
    kwargs['df_min'] = df_min
    kwargs['df_max'] = df_max

    # Load data.
    data = load_unsupervised_data(dataset)

    # Create object file names.
    fname_args = []
    fname_args.append(str(no_components))
    fname_args.append(fappend)
    mdl_fname = make_fname(METHOD, model, dataset, 'mdl', 'pk', *fname_args)
    vec_fname = make_fname(METHOD, model, dataset, 'vec', 'pk', *fname_args)
    dim_fname = make_fname(METHOD, model, dataset, 'dim', 'pk', *fname_args)
    mdl_path = os.path.join(MODEL_HOME,mdl_fname)
    vec_path = os.path.join(MODEL_HOME,vec_fname)
    dim_path = os.path.join(MODEL_HOME,dim_fname)

    # If we are specified to overwrite, or if required files missing,
    # train and store unsupervised components.
    model_files_present = os.path.isfile(mdl_path) and os.path.isfile(vec_path)

    if overwrite or not model_files_present:
        train(data, dataset, model, no_components, no_runs, **kwargs)

    # Evaluate model.
    eval(dataset, model, **kwargs)
