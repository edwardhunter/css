
from common import *
from sklearn.covariance import empirical_covariance
from numpy.linalg import cond
from data_utils import *
from matplotlib import pyplot as plt


dataset = '20news'
dims = range(2,81)

data = load_data(dataset)
data_train = data['train']
data_train_target = data['train_target']

vectorizer = TfidfVectorizer(stop_words='english')
x_train = vectorizer.fit_transform(data_train)

cond_nos = []

for dim in dims:
    print 'dims = %i' % dim
    fselector = SelectKBest(chi2, k=dim)
    x_train_r = fselector.fit_transform(x_train, data_train_target)
    x_train_r = normalize(x_train_r)
    x_train_r = x_train_r.toarray()
    sigma = empirical_covariance(x_train_r)
    cond_nos.append(cond(sigma))


plt.semilogy(dims, cond_nos)
plt.grid(axis='y',linestyle='--')
plt.xlabel('Feature dimension')
plt.ylabel('Covariance Matrix Condition Number')
plt.savefig('cov_cond.png')