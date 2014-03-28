#!/usr/bin/env python

# DBSCAN development in progress.


from common import *

# General parameters.
dataset = '20news4'
#dataset = 'reuters21578-10'
df_min = 1
df_max = 1.0

# DBSCAN parameters
eps = 0.5
min_samples=5
metric='euclidean'
algorithm='auto'
leaf_size=30
p=None
random_state=None

# Load data.
data = load_unsupervised_data(dataset)
_data = data['data']

# Vectorize data.
vectorizer = TfidfVectorizer(stop_words='english',sublinear_tf=True,
                min_df=df_min, max_df=df_max)
x = vectorizer.fit_transform(_data)


# Cluster data.
learner = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                 algorithm=algorithm, leaf_size=leaf_size, p=p,
                 random_state=random_state)
learner.fit(x)

# Get results
core_samples = learner.core_sample_indices_
labels = learner.labels_

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples = db.core_sample_indices_
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

