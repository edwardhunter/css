#!/usr/bin/env python

"""
@package css
@file css/spkm.py
@author Edward Hunter
@brief Spherical k-means core algorithm and support routines.
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


def count_cluster_sizes(no_clusters, no_docs, r):
    cluster_sizes = [0 for x in range(no_clusters)]
    for j in range(no_clusters):
        for i in range(no_docs):
            if r[i] == j:
                cluster_sizes[j] += 1
    return cluster_sizes


def initialize_clusters(x, cluster_ids, doc_ids):

    no_docs = len(doc_ids)
    no_clusters = len(cluster_ids)

    # Initialize shuffled docs ids.
    shuffled_docs = np.array(doc_ids)
    np.random.shuffle(shuffled_docs)

    # Initialize new assigments.
    r_new = np.zeros(no_docs, np.int32)

    # Initialize the centroids.
    mu = np.copy(x)[shuffled_docs[:no_clusters],:]

    return shuffled_docs, r_new, mu


def spkmeans(x, no_clusters, verbose=False, **kwargs):

    if not isinstance(x, np.ndarray):
        x = x.toarray()

    # Get doc and cluster ids.
    no_docs = x.shape[0]
    cluster_ids = range(no_clusters)
    doc_ids = range(no_docs)

    on_empty = kwargs.get('on_empty','restart')

    # Initialize the clusters
    shuffled_docs, r_new, mu = initialize_clusters(x, cluster_ids, doc_ids)

    # Initialize count and similary array.
    count = 0
    similarity = []

    # Spherical k means loop.
    while True:

        # Iteration start time.
        startime = time.time()

        # Update assignments.
        # Copy the old assignments.
        r = np.copy(r_new)

        # Compute the new assignments.
        products = np.dot(x,mu.T)
        r_new = np.argmax(products,axis=1)

        # Collect and sort the new scores.
        scores = np.array([products[i, r_new[i]] for i in doc_ids])
        scores_idx = np.argsort(scores)

        # Fix empty clusters here.
        empty = [i for i in cluster_ids if i not in r_new]
        if empty:
            if verbose:
                print 'Iteration %i: empty clusters: %s' % (count, str(empty))
            if on_empty == 'restart':
                if verbose:
                    print 'Reinitializing algorithm.'
                shuffled_docs, r_new, mu = initialize_clusters(x, cluster_ids,
                                                               doc_ids)
                continue

            else:
                if verbose:
                    print 'Reassinging remote data to empty clusters.'
                for i,j in enumerate(empty):
                    r_new[scores_idx[-(i+1)]] = j
                empty = [i for i in cluster_ids if i not in r_new]
                if verbose:
                    print 'Adjusted empty clusers: ' + str(empty)

        newsim = 0
        for i in range(no_docs):
            newsim += products[i,r_new[i]]
        similarity.append(newsim)

        # Exit if assigments do not change.
        if np.all(r == r_new):
            sizes = count_cluster_sizes(no_clusters, no_docs, r)
            return mu, r, similarity, sizes

        # Update centroids.
        mu = np.zeros_like(mu)
        for i in range(no_docs):
            mu[r_new[i],:] += x[i,:]
        for j in range(no_clusters):
            mu_norm = np.linalg.norm(mu[j,:])
            if mu_norm == 0.0:
                print 'Cluster %i still EMPTY!' % j

            else:
                mu[j,:] = mu[j,:]/mu_norm

        delta = time.time() - startime
        if verbose:
            print 'Iteration %i: %.2f seconds.' % (count, delta)
        count += 1

"""
no_docs = x_train.shape[0]
dim = x_train.shape[1]
print 'no docs: %i' % no_docs
print 'dimensionality: %i' % dim

similarities = []
cluster_sizes = []
mus = []
counts = []
r = []

for i in range(10):
    sim_i, count_i, r_i, mu_i = spkmeans(x_train, no_clusters, 'reassign')

    # Count the number in each cluster.
    cluster_sizes_i = count_cluster_sizes(no_clusters, no_docs, r_i)

    print 'Run %i: %i iterations, %f similarity, sizes: %s' %\
          (i, count_i, sim_i[-1], str(cluster_sizes_i))

    cluster_sizes.append(cluster_sizes_i)
    similarities.append(sim_i)
    mus.append(mu_i)
    counts.append(count_i)
    r.append(r_i)
"""


"""
# Plotting score functions.
for y in similarities:
    x = range(len(y))
    plt.plot(x,y,'k-')
    b = [0,y[-1]]
    a = [len(y)-1, len(y)-1]
    plt.plot(a,b,'r:')
    plt.xlabel('Total Similarity')
    plt.ylabel('Sorted Run')
plt.savefig('similarity_curves.png')
"""

"""
# Plotting sorted best scores for the ensemble.
best_sims = [x[-1] for x in similarities]
best_sims.sort(reverse=True)
x = range(len(best_sims))
plt.plot(x, best_sims, 'r-')
plt.xlabel('Total Similarity')
plt.ylabel('Sorted Run')
plt.show()
"""

"""
# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labels, rotation='vertical')
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.show()
"""


# Plotting cluster vectors.
"""
#plt.subplot(2,2,1)
plt.cla()
sort_idx = np.argsort(mus[0][0,:])[-10:]
sort_mu = mus[0][0,sort_idx]
x_ticks = []
x_labels = []
for i in range(len(sort_mu)):
    x = [sort_idx[i], sort_idx[i]]
    y = [0, 1.0]
    plt.plot(x,y,'r-')
y = mus[0][0,:]
x = range(len(y))
plt.plot(x,y,'k-')
plt.axis([0, len(y), 0, 1])
plt.xlabel('Term Feature')
plt.ylabel('TFIDF Score')
for i,idx in enumerate(sort_idx[::-1]):
    name = feature_names[feature_idx[idx]]
    plt.text(idx, 0.95-(i*0.035) ,name)
figpath = 'cluster1.png'
plt.savefig(figpath)

#plt.subplot(2,2,2)
plt.cla()
sort_idx = np.argsort(mus[0][1,:])[-10:]
sort_mu = mus[0][1,sort_idx]
for i in range(len(sort_mu)):
    x = [sort_idx[i], sort_idx[i]]
    y = [0, 1.0]
    plt.plot(x,y,'r-')
y = mus[0][1,:]
x = range(len(y))
plt.plot(x,y,'k-')
plt.axis([0, len(y), 0, 1])
plt.xlabel('Term Feature')
plt.ylabel('TFIDF Score')
for i,idx in enumerate(sort_idx[::-1]):
    name = feature_names[feature_idx[idx]]
    plt.text(idx, 0.95-(i*0.035) ,name)
figpath = 'cluster2.png'
plt.savefig(figpath)

#plt.subplot(2,2,3)
plt.cla()
sort_idx = np.argsort(mus[0][2,:])[-10:]
sort_mu = mus[0][2,sort_idx]
for i in range(len(sort_mu)):
    x = [sort_idx[i], sort_idx[i]]
    y = [0, 1.0]
    plt.plot(x,y,'r-')
y = mus[0][2,:]
x = range(len(y))
plt.plot(x,y,'k-')
plt.axis([0, len(y), 0, 1])
plt.xlabel('Term Feature')
plt.ylabel('TFIDF Score')
for i,idx in enumerate(sort_idx[::-1]):
    name = feature_names[feature_idx[idx]]
    plt.text(idx, 0.95-(i*0.035) ,name)
figpath = 'cluster3.png'
plt.savefig(figpath)

#plt.subplot(2,2,4)
plt.cla()
sort_idx = np.argsort(mus[0][3,:])[-10:]
sort_mu = mus[0][3,sort_idx]
for i in range(len(sort_mu)):
    x = [sort_idx[i], sort_idx[i]]
    y = [0, 1.0]
    plt.plot(x,y,'r-')
y = mus[0][3,:]
x = range(len(y))
plt.plot(x,y,'k-')
plt.axis([0, len(y), 0, 1])
plt.xlabel('Term Feature')
plt.ylabel('TFIDF Score')
for i,idx in enumerate(sort_idx[::-1]):
    name = feature_names[feature_idx[idx]]
    plt.text(idx, 0.95-(i*0.035) ,name)
figpath = 'cluster4.png'
plt.savefig(figpath)

#plt.show()
"""




"""
NOTES

feature_names = vectorizer.get_feature_names()
sort_idx = np.argsort(mu)
print 'Features: ' + str(len(feature_names))
print 'Mu shape: ' + str(mu.shape)
print 'sort_idx shape: ' + str(sort_idx.shape)
print 'Similarity: ' + str(similarity)

for j in range(no_clusters):
    print '-'*80
    print 'Top features for cluster %i' %j
    for i in range(20):
        print feature_names[sort_idx[j,i]]


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
        figpath = os.path.join(REPORT_HOME, figfname)
        plt.savefig(figpath)

for i in range(no_clusters):
    print np.linalg.norm(mus[0][i,:])


"""

"""
# Import common modules and utilities.
from common import *

#
no_clusters = 4
dim = 200

# Load training/testing utilities.
from data_utils import load_data, DATASETS

# Load and extract data.
data = load_data('20news4')
data_train = data['train']
data_train_target = data['train_target']


# Vectorize data.
# (10729, 99299) - 10K docs, 100K features
vectorizer = TfidfVectorizer(stop_words='english',sublinear_tf=True,
                min_df=10, max_df=0.9)
x_train = vectorizer.fit_transform(data_train)
feature_names = vectorizer.get_feature_names()
"""
