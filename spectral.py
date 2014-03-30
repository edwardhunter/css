#!/usr/bin/env python

# DBSCAN development in progress.


from common import *

# Load training/testing utilities.
from data_utils import load_unsupervised_data, DATASETS


# General parameters.
#dataset = '20news4'
dataset = 'reuters21578-10'
df_min = 10
df_max = 0.8


# Load data.
data = load_unsupervised_data(dataset)
_data = data['data']
_target = data['target']




# Vectorize data.
print 'Vectorizing...'
vectorizer = TfidfVectorizer(stop_words='english',sublinear_tf=True,
                min_df=df_min, max_df=df_max)
x = vectorizer.fit_transform(_data)
feature_names = vectorizer.get_feature_names()
print x.shape
no_docs = x.shape[0]
no_dims = x.shape[1]


print 'Spectral clustering...'
n_clusters = 10
learner = SpectralClustering(n_clusters=n_clusters, gamma=0.01, affinity='rbf')
learner.fit(x)


labels = learner.labels_
print len(set(labels))
print set(labels)
for i in range(n_clusters):
    print np.argwhere(labels==i).shape

for i in range(n_clusters):
    lbl_idx = np.argwhere(labels==i)
    c = x[lbl_idx[:,0],:]
    print type(c)
    mu = np.sum(c.toarray(), axis=0)
    mu = mu / np.linalg.norm(mu)
    sort_idx = np.argsort(mu)[-10:]
    sort_mu = mu[sort_idx]

    plt.cla()
    for j in range(sort_mu.shape[0]):
        xx = [sort_idx[j], sort_idx[j]]
        y = [0, 1]
        plt.plot(xx,y,'r-')
    xx = range(mu.shape[0])
    plt.plot(xx,mu,'k-')
    plt.axis([0, len(mu), 0, 1])
    plt.xlabel('Term Feature')
    plt.ylabel('TFIDF Score')
    title_str = 'Tree Hierarchy Cluster'
    plt.title(title_str)
    for k, idx in enumerate(sort_idx[::-1]):
        name = feature_names[idx]
        plt.text(idx, 0.95-(k*0.035), name)
    fname = 'cluster_%i.png' % i
    plt.savefig(fname)


"""
Signature:
SpectralClustering(n_clusters=8, eigen_solver=None, random_state=None, n_init=10,
    gamma=1.0, affinity='rbf', n_neighbors=10, k=None, eigen_tol=0.0,
    assign_labels='kmeans', mode=None, degree=3, coef0=1, kernel_params=None

SpectralClustering(n_clusters=8, gamma=1.0, affinity='rbf')



"""



"""
Signature:
AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True,
    preference=None, affinity='euclidean', verbose=False)

# Affinity propagation clustering

# Compute distance matrix in Radians on the unit hypersphere.
# Take care to clip floating errors giving values > 1.0.
# Affinity should be nonpositive.
print 'Computing document distance matrix...'
d = x.dot(x.T)
d[d.ceil()>1] = 1
d = - ((math.pi/2.0) - d.arcsin().toarray())

print 'Affinity clustering...'
learner = AffinityPropagation(affinity='precomputed', verbose=True)
learner.fit(d)
labels = learner.labels_
cluters = learner.cluster_center_indices_

"""




"""
# Ward Clustering.
print 'Clustering...'
n_clusters = 10
labels =np.zeros(shape=(no_docs,))
learner = Ward(n_clusters=n_clusters, compute_full_tree=False)
x = x.toarray()
learner.fit(x)
labels = learner.labels_

print labels.shape
print set(labels)
for i in range(n_clusters):
    print np.argwhere(labels==i).shape[0]

"""


"""
# Denisty clustering.

# Compute distance matrix in Radians on the unit hypersphere.
# Take care to clip floating errors giving values > 1.0.
print 'Computing document distance matrix...'
d = x.dot(x.T)
d[d.ceil()>1] = 1
d = (math.pi/2.0) - d.arcsin().toarray()

# Cluster data.
print 'Clustering...'
learner = DBSCAN(eps=.3, min_samples=5, metric='precomputed')
learner.fit(d)
core_samples = learner.core_sample_indices_
labels = learner.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print 'Done. Found %i clusters.' % n_clusters
print str(set(labels))
print labels.shape
#print core_samples.shape
for i in range(n_clusters):
    print np.argwhere(labels==i).shape
print np.argwhere(labels==-1).shape
"""



"""
for i in range(n_clusters):
    lbl_idx = np.argwhere(labels==i)
    c = x[lbl_idx[:,0],:]
    mu = np.sum(c,0)
    mu = mu / np.linalg.norm(mu)
    sort_idx = np.argsort(mu)[-10:]
    sort_mu = mu[sort_idx]

    plt.cla()
    for j in range(sort_mu.shape[0]):
        xx = [sort_idx[j], sort_idx[j]]
        y = [0, 1]
        plt.plot(xx,y,'r-')
    xx = range(mu.shape[0])
    plt.plot(xx,mu,'k-')
    plt.axis([0, len(mu), 0, 1])
    plt.xlabel('Term Feature')
    plt.ylabel('TFIDF Score')
    title_str = 'Tree Hierarchy Cluster'
    plt.title(title_str)
    for k, idx in enumerate(sort_idx[::-1]):
        name = feature_names[idx]
        plt.text(idx, 0.95-(k*0.035), name)
    fname = 'cluster_%i.png' % i
    plt.savefig(fname)
"""