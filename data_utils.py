#!/usr/bin/env python

"""
@package css
@file css/data_utils.py
@author Edward Hunter
@brief Utility module for retrieving and loading training and testing text datasets.
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

DATASETS = ('20news','20news4', '20news5', 'reuters21578-10', 'senate_3_3')

"""
20 Newsgroups Labels
0   'alt.atheism'
1   'comp.graphics'
2   'comp.os.ms-windows.misc'
3   'comp.sys.ibm.pc.hardware'
4   'comp.sys.mac.hardware'
5   'comp.windows.x'
6   'misc.forsale'
7   'rec.autos'
8   'rec.motorcycles'
9   'rec.sport.baseball'
10  'rec.sport.hockey'
11  'sci.crypt'
12  'sci.electronics'
13  'sci.med'
14  'sci.space'
15  'soc.religion.christian'
16  'talk.politics.guns'
17  'talk.politics.mideast'
18  'talk.politics.misc'
19  'talk.religion.misc'

4 Newgroups Labels:
1   'comp.graphics'
7   'rec.autos'
14  'sci.space'
19  'talk.religion.misc'

5 Newsgroups Labels:
0   computers
    1 'comp.graphics'
    2 'comp.os.ms-windows.misc'
    3 'comp.sys.ibm.pc.hardware'
    4 'comp.sys.mac.hardware'
    5 'comp.windows.x'
1   recreation
    7   'rec.autos'
    8   'rec.motorcycles'
    9   'rec.sport.baseball'
    10  'rec.sport.hockey'
2   science
    11  'sci.crypt'
    12  'sci.electronics'
    13  'sci.med'
    14  'sci.space'
3   politics
    16  'talk.politics.guns'
    17  'talk.politics.mideast'
    18  'talk.politics.misc'
4   religion
    0   'alt.atheism'
    15  'soc.religion.christian'
    19  'talk.religion.misc'
"""


def download_20news(data_home=DATA_HOME, news_home=NEWS_HOME,
                    news_cache_name=NEWS_CACHE_NAME):
    """
    Download the full unfiltered 20 newsgroups data cache.
    @param data_home: relative directory name for data.
    @param news_home: directory name inside data_home to hold a temporary
        new archive. Directory deleted when done.
    @param news_cache_name: Name of the 20news cache pickle left in data_home.
    """

    print 'Downloading dataset 20 Newsgroups.'
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

    # Populate the 20 newsgroup data into our result data dictionary.
    data = {}
    data['target_names'] = list(news_data_train.target_names)
    data['train'] = list(news_data_train.data)
    data['test'] = list(news_data_test.data)
    data['train_target'] = list(news_data_train.target)
    data['test_target'] = list(news_data_test.target)

    # Write out a zipped pickle for the full 20 news set.
    news20_path = os.path.join(data_home, NEWS20_CACHE_NAME)
    open(news20_path, 'wb').write(pickle.dumps(data).encode('zip'))

    # Populate the 4 newsgroup data into our result data dictionary.
    data4_cats = {1:0, 7:1, 14:2, 19:3}
    data4 = {
        'target_names' : [data['target_names'][x] for x in data4_cats.keys()],
        'train' : [],
        'test' : [],
        'train_target' : [],
        'test_target' : []
    }
    for i,x in enumerate(data['train_target']):
        if x in data4_cats.keys():
            data4['train'].append(data['train'][i])
            data4['train_target'].append(data4_cats[data['train_target'][i]])
    for i,x in enumerate(data['test_target']):
        if x in data4_cats.keys():
            data4['test'].append(data['test'][i])
            data4['test_target'].append(data4_cats[data['test_target'][i]])

    # Write out a zipped pickle for the 4 news set.
    news20_path = os.path.join(data_home, NEWS20_4_CACHE_NAME)
    open(news20_path, 'wb').write(pickle.dumps(data4).encode('zip'))

    # Populate the 5 newsgroup data into our result data dictionary.
    data5_cats = {0:4, 1:0, 2:0, 3:0, 4:0, 5:0, 7:1, 8:1, 9:1, 10:1, 11:2,
        12:2, 13:2, 14:2, 15:4, 16:3, 17:3, 18:3, 19:4}
    data5 = {
        'target_names' : ['computers','recreation','science',
                          'politics','religion'],
        'train' : [],
        'test' : [],
        'train_target' : [],
        'test_target' : []
    }
    for i,x in enumerate(data['train_target']):
        if x in data5_cats.keys():
            data5['train'].append(data['train'][i])
            data5['train_target'].append(data5_cats[data['train_target'][i]])
    for i,x in enumerate(data['test_target']):
        if x in data5_cats.keys():
            data5['test'].append(data['test'][i])
            data5['test_target'].append(data5_cats[data['test_target'][i]])

    # Write out a zipped pickle for the 5 news set.
    news20_path = os.path.join(data_home, NEWS20_5_CACHE_NAME)
    open(news20_path, 'wb').write(pickle.dumps(data5).encode('zip'))


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

def get_cat_data(name, data_home=DATA_HOME):
    """
    Return a dict of data organized for inspection.
    @param name: The data file name.
    @data_home: The data file directory.
    @return cat_data: A dict of data organized by category.
    """
    data = load_data(name, data_home)
    cat_data = {'train' : {}, 'test' : {}}
    for x in data['target_names']:
        cat_data['train'][x] = []
        cat_data['test'][x] = []
    for i, x in enumerate(data['train_target']):
        cat_data['train'][data['target_names'][x]].append(data['train'][i])
    for i, x in enumerate(data['test_target']):
        cat_data['test'][data['target_names'][x]].append(data['test'][i])
    return cat_data


def load_data(name, data_home=DATA_HOME):
    """
    Load a data pickle into memory for processing.
    @param name: Name of the pickle less the .pkz extension.
    @param data_home: Directory to find the pickle. Default is ./data.
    @retrun data: The supervised data dictionary.
    """
    file_path = os.path.join(data_home, name + '.pkz')
    print 'Loading: ' + file_path
    if not os.path.isfile(file_path):
        raise ValueError('Could not find the file %s' % file_path)

    data = pickle.loads(open(file_path, 'rb').read().decode('zip'))
    print 'Class Names:'
    training_total = 0
    testing_total = 0
    for i, x in enumerate(data['target_names']):
        training_count = len([y for y in data['train_target'] if y==i])
        testing_count = len([y for y in data['test_target'] if y==i])
        training_total += training_count
        testing_total += testing_count
        print '%5i  %25s  train size: %8i, test size: %8i' % \
              (i, x, training_count, testing_count)
    print '%5s  %25s  train size: %8i, test size: %8i' % \
            ('', 'Totals:', training_total, testing_total)

    return data


def load_unsupervised_data(name, data_home=DATA_HOME):
    """
    Load a data pickle into memory and collapse training and testing
    into one dataset.
    @param name: Name of the pickle less the .pkz extension.
    @param data_home: Directory to find the pickle. Default is ./data.
    @return data: The unsupervised data dictionary.
    """
    data = load_data(name, data_home)
    _data = list(data['train'])
    _data.extend(list(data['test']))
    _target = list(data['train_target'])
    _target.extend(list(data['test_target']))
    combined_data = dict(
        data=_data,
        target=_target,
        target_names=data['target_names']
    )
    print '\nLoaded for unsupervised learning, total data size: %i.' % len(_data)

    return combined_data


# If run as a script, destroy and recreate all data pickles.
if __name__ == '__main__':

    # Parse command line arguments and options.
    usage = 'usage: %prog [options]'
    description = 'Download and construct training and testing data.'
    p = optparse.OptionParser(usage=usage, description=description)
    p.add_option('-d','--directory', action='store', dest='data_home',
                 help='Home directory for data file (default: %s).' % DATA_HOME)
    p.set_defaults(data_home=DATA_HOME)

    (opts, args) = p.parse_args()

    data_home = opts.data_home
    print str(data_home)
    if os.path.isdir(data_home):
        shutil.rmtree(data_home)

    download_20news(data_home)
    make_20news(data_home)
    download_reuters(data_home)
    split_reuters10(data_home)
    make_reuters10(data_home)

