#!/usr/bin/env python

# DBSCAN development in progress.


from common import *
from vis_util import *

# Load training/testing utilities.
from data_utils import load_unsupervised_data, DATASETS

# General parameters.
dataset = '20news4'
#dataset = 'reuters21578-10'
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
n_clusters = 6
learner = SpectralClustering(n_clusters=n_clusters, gamma=0.01, affinity='rbf')
learner.fit(x)
labels = learner.labels_


if not os.path.exists(REPORT_HOME):
        os.makedirs(REPORT_HOME)

cluster_silhouettes(x, labels, 'Cluster', 'spectral', dataset)



"""
Signature:
SpectralClustering(n_clusters=8, eigen_solver=None, random_state=None, n_init=10,
    gamma=1.0, affinity='rbf', n_neighbors=10, k=None, eigen_tol=0.0,
    assign_labels='kmeans', mode=None, degree=3, coef0=1, kernel_params=None

SpectralClustering(n_clusters=8, gamma=1.0, affinity='rbf')



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

