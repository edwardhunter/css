#!/usr/bin/env python

"""
@package css.data_utils
@file css/data_utils.py
@author Edward Hunter
@author K Sree Harsha
@brief Utility module for retrieving and loading training and testing
text datasets.
"""

from common import *
from reuters_parser import ReutersParser

# Default data directory.
DATA_HOME = os.path.join('.', 'data')

# 20 newsgroups defaults
NEWS_CACHE_NAME = "20news-bydate.pkz"
NEWS20_CACHE_NAME = "20news.pkz"
NEWS20_4_CACHE_NAME = "20news4.pkz"
NEWS20_5_CACHE_NAME = "20news5.pkz"
NEWS_HOME = "20news_home"
NEWS_ALL_CATEGORIES = None
NEWS_REMOVE = ('headers', 'footers', 'quotes')

# Reuters defaults.
REUTERS_URL = ("http://kdd.ics.uci.edu/databases/reuters21578/"
                "reuters21578.tar.gz")
REUTERS_HOME = "reuters_home"
REUTERS_ARCHIVE_NAME = "reuters21578.tar.gz"
REUTERS_CACHE_NAME = "reuters21578.pkz"
REUTERS10_IDX_NAME = "reuters21578-10-idx.pkz"
REUTERS10_CACHE_NAME = "reuters21578-10.pkz"

DATASETS = ('20news','reuters21578-10')

def download_20news(data_home=DATA_HOME, news_home=NEWS_HOME,
                    news_cache_name=NEWS_CACHE_NAME):
    """
    Download the full unfiltered 20 newsgroups data cache.
    @param data_home: relative directory name for data.
    @param news_home: directory name inside data_home to hold a temporary
        new archive. Directory deleted when done.
    @param news_cache_name: Name of the 20news cache pickle left in data_home.
    """

    cache_path = os.path.join(data_home, news_cache_name)
    twenty_home = os.path.join(data_home, news_home)
    download_20newsgroups(target_dir=twenty_home, cache_path=cache_path)


def make_20news(data_home=DATA_HOME):
    """
    Make pickles for all the 20newsgroups datasets.
    @param data_home: relative directory name for data.
    """

    # Fetch the 20 newsgroup data, removing headers, footers, quotes.
    categories = None
    remove = ('headers', 'footers', 'quotes')
    
    news_data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove, data_home=data_home)

    news_data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove, data_home=data_home)
    
    news4_data_train = fetch_20news4groups(subset='train',
                                shuffle=True, random_state=42,
                                remove=remove, data_home=data_home)

    news4_data_test = fetch_20news4groups(subset='test',
                               shuffle=True, random_state=42,
                               remove=remove, data_home=data_home)
    
    news5_data_train = fetch_20news5groups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove, data_home=data_home)

    news5_data_test = fetch_20news5groups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove, data_home=data_home)
	
    # Populate the 20 newsgroup data into our result data dictionary.
    data = {}
    data['target_names'] = news_data_train.target_names
    data['train'] = news_data_train.data
    data['test'] = news_data_test.data
    data['train_target'] = news_data_train.target
    data['test_target'] = news_data_test.target

    # Write out a zipped pickle for the full 20 news set.
    news20_path = os.path.join(data_home, NEWS20_CACHE_NAME)
    open(news20_path, 'wb').write(pickle.dumps(data).encode('zip'))

    data = {}
    data['target_names'] = news4_data_train.target_names
    data['train'] = news4_data_train.data
    data['test'] = news4_data_test.data
    data['train_target'] = news4_data_train.target
    data['test_target'] = news4_data_test.target

    # Write out a zipped pickle for the full 20 news set.
    news20_path = os.path.join(data_home, NEWS20_4_CACHE_NAME)
    open(news20_path, 'wb').write(pickle.dumps(data).encode('zip'))

    data = {}
    data['target_names'] = news5_data_train.target_names
    data['train'] = news5_data_train.data
    data['test'] = news5_data_test.data
    data['train_target'] = news5_data_train.target
    data['test_target'] = news5_data_test.target

    # Write out a zipped pickle for the full 20 news set.
    news20_path = os.path.join(data_home, NEWS20_5_CACHE_NAME)
    open(news20_path, 'wb').write(pickle.dumps(data).encode('zip'))


def download_reuters(data_home=DATA_HOME, reuters_home=REUTERS_HOME,
                     reuters_cache_name=REUTERS_CACHE_NAME):
    """
    Download the full unfiltered reuters data cache.
    @param data_home: relative directory name for data.
    @param reuters_home: directory name inside data_home to hold a temporary
        new archive. Directory deleted when done.
    @param reuters_cache_name: Name of the reuters cache pickle left in data_home.
    """

    archive_dir = os.path.join(data_home, reuters_home)
    archive_path = os.path.join(archive_dir, REUTERS_ARCHIVE_NAME)
    cache_path = os.path.join(data_home, reuters_cache_name)

    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)

    if not os.path.exists(archive_path):
        print "Downloading dataset from %s (8.2 MB)" % REUTERS_URL
        opener = urlopen(REUTERS_URL)
        open(archive_path, 'wb').write(opener.read())

    print "Decompressing %s" % archive_path
    tarfile.open(archive_path, "r:gz").extractall(path=archive_dir)
    os.remove(archive_path)

    p = ReutersParser()
    archive_dir = os.path.join(data_home, reuters_home)
    for i in range(22):
        fname = 'reut2-%03d.sgm' % i
        fpath = os.path.join(archive_dir, fname)
        print 'Reading file: %s' % fpath
        p.feed(open(fpath).read())
    p.close()

    # Write out a zipped pickle for the full reuters corpus parser.
    open(cache_path, 'wb').write(pickle.dumps(p).encode('zip'))

    shutil.rmtree(archive_dir)

    return p


def split_reuters10(data_home=DATA_HOME, reuters_cache_name=REUTERS_CACHE_NAME,
                    reuters10_idx_name=REUTERS10_IDX_NAME, test_frac=0.2):
    """
    Create a pickle for split training and testing data containing document ids and
    labels only.
    @param data_home: relative directory name for data.
    @param reuters_cache_name: Name of the reuters cache pickle in data_home.
    @param reuters10_idx_name: Name of the training index pickle left in data_home.
    @param test_frac: Float fraction of topic documents to use for test set.
    """
    reuters_cache_path = os.path.join(data_home, reuters_cache_name)
    p = pickle.loads(open(reuters_cache_path, 'rb').read().decode('zip'))
    data = p.create_test_split(test_frac=test_frac)

    # Write out a zipped pickle for the full reuters corpus parser.
    reuters10_idx_path = os.path.join(data_home, reuters10_idx_name)
    open(reuters10_idx_path, 'wb').write(pickle.dumps(data).encode('zip'))


def make_reuters10(data_home=DATA_HOME, reuters_cache_name=REUTERS_CACHE_NAME,
                reuters10_idx_name=REUTERS10_IDX_NAME,
                reuters10_cache_name=REUTERS10_CACHE_NAME):
    """
    Create a training-testing reuters data pickle with document bodies.
    @param data_home: relative directory name for data.
    @param reuters_cache_name: Name of the reuters cache pickle in data_home.
    @param reuters10_idx_name: Name of the training index pickle in data_home.
    @param reuters10_cache_name: Name of fully populated pickle left in data_home.
    """

    reuters_cache_path = os.path.join(data_home, reuters_cache_name)
    reuters10_idx_path = os.path.join(data_home, reuters10_idx_name)
    reuters10_cache_path = os.path.join(data_home, reuters10_cache_name)

    # Load the full corpus parser and the reuters 10 index dictionary.
    p = pickle.loads(open(reuters_cache_path, 'rb').read().decode('zip'))
    data = pickle.loads(open(reuters10_idx_path, 'rb').read().decode('zip'))

    # Populate the index dictionary.
    data['train'] = []
    for id in data['train_id']:
        data['train'].append(p.corpus[id]['body'])

    data['test'] = []
    for id in data['test_id']:
        data['test'].append(p.corpus[id]['body'])

    # Write out the populated reuters 10 dictionary.
    open(reuters10_cache_path, 'wb').write(pickle.dumps(data).encode('zip'))


def load_data(name, data_home=DATA_HOME):
    """
    Load a data pickle into memory for processing.
    @param name: Name of the pickle less the .pkz extension.
    @param data_home: Directory to find the pickle. Default is ./data.
    """
    file_path = os.path.join(data_home, name + '.pkz')
    if not os.path.isfile(file_path):
        raise ValueError('Could not find the file %s' % file_path)

    return pickle.loads(open(file_path, 'rb').read().decode('zip'))


# If run as a script, destroy and recreate all data pickles.
if __name__ == '__main__':

    if len(sys.argv) > 1:
        data_home = sys.argv[1]
    else:
        data_home=DATA_HOME

    if os.path.isdir(data_home):
        shutil.rmtree(data_home)

    download_20news(data_home)
    make_20news(data_home)
    download_reuters(data_home)
    split_reuters10(data_home)
    make_reuters10(data_home)

