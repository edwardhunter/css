"""Caching loader for the 20 newsgroups text classification dataset


The description of the dataset is available on the official website at:

    http://people.csail.mit.edu/jrennie/20Newsgroups/

Quoting the introduction:

    The 20 Newsgroups data set is a collection of approximately 20,000
    newsgroup documents, partitioned (nearly) evenly across 20 different
    newsgroups. To the best of my knowledge, it was originally collected
    by Ken Lang, probably for his Newsweeder: Learning to filter netnews
    paper, though he does not explicitly mention this collection. The 20
    newsgroups collection has become a popular data set for experiments
    in text applications of machine learning techniques, such as text
    classification and text clustering.

This dataset loader will download the recommended "by date" variant of the
dataset and which features a point in time split between the train and
test sets. The compressed dataset size is around 14 Mb compressed. Once
uncompressed the train set is 52 MB and the test set is 34 MB.

The data is downloaded, extracted and cached in the '~/scikit_learn_data'
folder.

The `fetch_20newsgroups` function will not vectorize the data into numpy
arrays but the dataset lists the filenames of the posts and their categories
as target labels.

The `fetch_20newsgroups_tfidf` function will in addition do a simple tf-idf
vectorization step.

"""


import os
import logging
import tarfile
import pickle
import shutil
import re
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import *
from sklearn.utils import *
from sklearn.utils.fixes import *
from sklearn.externals import *

if six.PY3:
    from urllib.request import urlopen
else:
    from urllib2 import urlopen


logger = logging.getLogger(__name__)


URL = ("http://people.csail.mit.edu/jrennie/"
       "20Newsgroups/20news-bydate.tar.gz")
ARCHIVE_NAME = "20news-bydate.tar.gz"
CACHE_NAME = "20news-bydate.pkz"
TRAIN_FOLDER = "20news-bydate-train"
TEST_FOLDER = "20news-bydate-test"


def download_20newsgroups(target_dir, cache_path):
    """Download the 20 newsgroups data and stored it as a zipped pickle."""
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.exists(archive_path):
        logger.warn("Downloading dataset from %s (14 MB)", URL)
        opener = urlopen(URL)
        open(archive_path, 'wb').write(opener.read())

    logger.info("Decompressing %s", archive_path)
    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)
    os.remove(archive_path)

    # Store a zipped pickle
    cache = dict(train=load_files(train_path, encoding='latin1'),
                 test=load_files(test_path, encoding='latin1'))
    open(cache_path, 'wb').write(pickle.dumps(cache).encode('zip'))
    shutil.rmtree(target_dir)
    return cache


def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')
    return after


_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    """
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.

    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text


def fetch_20news4groups(data_home=None, subset='train',
                       shuffle=True, random_state=42,
                       remove=(),
                       download_if_missing=True):
    """Load the filenames and data from the 20 newsgroups dataset.

    Parameters
    ----------
    subset: 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    data_home: optional, default: None
        Specify an download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.


    shuffle: bool, optional
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    random_state: numpy random number generator or seed integer
        Used to shuffle the dataset.

    download_if_missing: optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    remove: tuple
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

        'headers' follows an exact standard; the other filters are not always
        correct.
    """

    data_home = get_data_home(data_home=data_home)
    cache_path = os.path.join(data_home, CACHE_NAME)
    twenty_home = os.path.join(data_home, "20news_home")
    cache = None
    categories = [ 'alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space', ]
    if os.path.exists(cache_path):
        try:
            cache = pickle.loads(open(cache_path, 'rb').read().decode('zip'))
        except Exception as e:
            print(80 * '_')
            print('Cache loading failed')
            print(80 * '_')
            print(e)

    if cache is None:
        if download_if_missing:
            cache = download_20newsgroups(target_dir=twenty_home,
                                          cache_path=cache_path)
        else:
            raise IOError('20Newsgroups dataset not found')

    if subset in ('train', 'test'):
        data = cache[subset]
    elif subset == 'all':
        data_lst = list()
        target = list()
        filenames = list()
        for subset in ('train', 'test'):
            data = cache[subset]
            data_lst.extend(data.data)
            target.extend(data.target)
            filenames.extend(data.filenames)

        data.data = data_lst
        data.target = np.array(target)
        data.filenames = np.array(filenames)
        data.description = 'the 20 newsgroups by date dataset'
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    if 'headers' in remove:
        data.data = [strip_newsgroup_header(text) for text in data.data]
    if 'footers' in remove:
        data.data = [strip_newsgroup_footer(text) for text in data.data]
    if 'quotes' in remove:
        data.data = [strip_newsgroup_quoting(text) for text in data.data]

    
    labels = [(data.target_names.index(cat), cat) for cat in categories]
        # Sort the categories to have the ordering of the labels
    labels.sort()
    labels, categories = zip(*labels)
    mask = in1d(data.target, labels)
    data.filenames = data.filenames[mask]
    data.target = data.target[mask]
        # searchsorted to have continuous labels
    data.target = np.searchsorted(labels, data.target)
    data.target_names = list(categories)
        # Use an object array to shuffle: avoids memory copy
    data_lst = np.array(data.data, dtype=object)
    data_lst = data_lst[mask]
    data.data = data_lst.tolist()

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()

    return data



