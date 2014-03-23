#!/usr/bin/env python

"""
@package css
@file css/reuters_parser.py
@author Edward Hunter
@brief Class for parsing SGML Reuters-21578 data.
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
import sgmllib

DATASETS = ('20news')

class ReutersParser(sgmllib.SGMLParser):
    """
    Class for parsing SGML Reuters-21578 data.
    """

    # The top most frequent single topic categories.
    top_10 = [
        'earn', # 3945
        'acq', # 2362
        'crude', # 408
        'trade', # 362
        'money-fx', # 307
        'interest', # 285
        'money-supply', # 161
        'ship', # 158
        'sugar', # 143
        'coffee' # 116
    ]

    def __init__(self):
        """
        Initialize members.
        """
        sgmllib.SGMLParser.__init__(self)
        self.reuters = False
        self.topics = False
        self.title = False
        self.body = False
        self.body_text = ''
        self.topic_set = set()
        self.pattern1 = re.compile('[()\"\<>*]')
        self.pattern2 = re.compile('Blah|blah')
        self.pattern3 = re.compile('[^\x20-\x7E]')
        self.pattern4 = re.compile('\s{2,}')
        self.corpus = {}
        self.single_topic_corpus = {}
        self.all_topics = set()
        self.single_topics = set()

    def start_reuters(self, attributes):
        """
        Begin a reuters tag that delimits a document.
        """
        self.body_text = ''
        self.topic_set = set()
        attr_dict = dict(attributes)
        self.id = int(attr_dict['newid'])

    def end_reuters(self):
        """
        Ends a reuters tag that delimits a document.
        """
        #self.body_text = self.body_text.replace('<','')
        #self.body_text = self.body_text.replace('>','')
        self.body_text = self.pattern1.sub('', self.body_text)
        self.body_text = self.pattern2.sub('', self.body_text)
        self.body_text = self.pattern3.sub('', self.body_text)
        self.body_text = self.pattern4.sub(' ', self.body_text)
        self.corpus[self.id] = {
            'topics' : copy.deepcopy(self.topic_set),
            'body' : copy.deepcopy(self.body_text)
        }
        if len(self.topic_set) == 1:
            topic = self.topic_set.pop()
            self.single_topics.add(topic)
            if topic not in self.single_topic_corpus:
                self.single_topic_corpus[topic] = []
            self.single_topic_corpus[topic].append((self.id,
                copy.deepcopy(self.body_text)))

    def start_title(self, attributes):
        """
        Begins a title.
        """
        self.title = True

    def end_title(self):
        """
        Ends a title
        """
        self.title = False

    def start_body(self, attributes):
        """
        Begins body.
        """
        self.body = True

    def end_body(self):
        """
        Ends body.
        """
        self.body = False

    def start_topics(self, attributes):
        """
        Begins topics.
        """
        self.topics = True

    def end_topics(self):
        """
        Ends topics.
        """
        self.topics = False

    def handle_data(self, data):
        """
        Process data blocks.
        """
        if self.topics:
            self.topic_set.add(data)
            self.all_topics.add(data)

        if self.body or self.title:
            if len(self.body_text)>0:
                self.body_text += ' '
            self.body_text += data

    def dump_corpus(self):
        """
        Print out the entire corpus.
        """
        for k,v in self.corpus.iteritems():
            print '-'*80
            print 'ID: %i' % k
            print 'TOPICS:'
            for x in v['topics']:
                print x
            print 'BODY:'
            print v['body']

    def dump_single_topic_corpus(self):
        """
        Print out the single topic corpus.
        """
        for k,v in self.single_topic_corpus.iteritems():
            print '-'*80
            print 'topic: %s' % k
            print '-'*80
            for x in v:
                print 'ID: %i' % x[0]
                print 'BODY:'
                print x[1]
                print

    def dump_st_dist(self):
        """
        Print out the ordered single topic frequency distribution.
        """
        dist = {}
        for k, v in self.single_topic_corpus.iteritems():
            count = len(v)
            dist[k] = count
        sorted_keys = sorted(dist, key=dist.get, reverse=True)
        for k in sorted_keys:
            print '%s = %i' % (k, dist[k])

    def create_test_split(self, test_frac=0.2):
        """
        Create a random test-train split.
        @param test_frac: A float fraction of documents to use as test data.
        @return data: dictionary containing single topic training-testing split
            without the text body data; text indexes only.
        """
        data = {}
        data['target_names'] = copy.deepcopy(ReutersParser.top_10)
        data['train'] = []
        data['test'] = []
        data['train_target'] = []
        data['test_target'] = []
        data['train_id'] = []
        data['test_id'] = []
        for i, k in enumerate(ReutersParser.top_10):
            val = self.single_topic_corpus[k]
            ids = [x[0] for x in val]
            random.shuffle(ids)
            test_count = int(round(len(ids)*test_frac))
            train_count = len(ids) - test_count
            train_ids = ids[:train_count]
            test_ids = ids[train_count:]
            for x in train_ids:
                #data['train'].append(self.corpus[x])
                data['train_target'].append(i)
                data['train_id'].append(x)
            for x in test_ids:
                #data['test'].append(self.corpus[x])
                data['test_target'].append(i)
                data['test_id'].append(x)

        return data
