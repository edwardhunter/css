#!/usr/bin/env python

"""
@package css
@file css/vis_utils.py
@author Edward Hunter
@brief Visualization utilities.
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


def cluster_vectors(mu, no_components, sizes, feature_names, no_top, method,
                    model, dataset):
    """
    # Plot cluster vectors.
    @param mu: xx.
    @param no_components: xx.
    @param sizes: xx.
    @param feature_names: xx.
    @param no_top: xx.
    @param method: xx.
    @param model: xx.
    @param dataset: xx.
    """
    for i in range(no_components):
        plt.cla()
        sort_idx = np.argsort(mu[i,:])[-no_top:]
        sort_mu = mu[i,sort_idx]
        for j in range(len(sort_mu)):
            x = [sort_idx[j], sort_idx[j]]
            y = [0, 1]
            plt.plot(x,y,'r:')
        y = mu[i,:]
        x = range(len(y))
        plt.plot(x,y,'k-')
        plt.axis([0, len(y), 0, 1])
        plt.xlabel('Cluster %i Vector Terms' % i)
        plt.ylabel('TFIDF Score')
        for k, idx in enumerate(sort_idx[::-1]):
            name = feature_names[idx]
            sha = 'right' if (idx > .9*x[-1]) else 'left'
            plt.text(idx, 0.95-(k*0.035), name, horizontalalignment=sha)
        plt.tight_layout()
        suffix = 'cluster_%i_vector' % i
        fname = make_fname(method, model, dataset, suffix, 'png')
        fpath = os.path.join(REPORT_HOME,fname)
        plt.savefig(fpath)


def cluster_ensesmble_similarities(similarities, no_components, method, model, dataset):
    """
    # Plot sorted best similarities for the ensemble.
    @param similarities: xx.
    @param no_components: xx.
    @param method: xx.
    @param model: xx.
    @param dataset: xx.
    """
    plt.cla()
    best_sims = [x[-1] for x in similarities]
    best_sims.sort(reverse=True)
    x = range(len(best_sims))
    plt.plot(x, best_sims, 'r-')
    plt.xlabel('Sorted Runs')
    plt.ylabel('Total Cosine Similarity')
    plt.tight_layout()
    suffix = '%i_clusters_ensemble_similarities' % no_components
    fname = make_fname(method, model, dataset, suffix, 'png')
    fpath = os.path.join(REPORT_HOME,fname)
    plt.savefig(fpath)


def cluster_sim_curves(similarities, no_components, method, model, dataset):
    """
    Plot similarity curves.
    @param similarities: xx.
    @param no_components: xx.
    @param method: xx.
    @param model: xx.
    @param dataset: xx.
    """
    plt.cla()
    for y in similarities:
        x = range(len(y))
        plt.plot(x,y,'k-')
        b = [0,y[-1]]
        a = [len(y)-1, len(y)-1]
        plt.plot(a,b,'r:')
    plt.xlabel('Iterations, %i Runs' % len(similarities))
    plt.ylabel('Total Cosine Similarity')
    plt.tight_layout()
    suffix = '%i_clusters_simcurves' % no_components
    fname = make_fname(method, model, dataset, suffix, 'png')
    fpath = os.path.join(REPORT_HOME,fname)
    plt.savefig(fpath)


def cluster_silhouettes(x, mu, labels, method, model, dataset, project=True):
    """
    Plot cluster silhouettes.
    @param x: xx.
    @param labels: xx.
    @param method: xx.
    @param model: xx.
    @param dataset: xx.
    @param project: xx.
    """

    if project:
        x, labels = project_data(x, mu, labels)

    no_components = len(set(labels))

    sil = silhouette_samples(x, labels)
    #sil = silhouette_score(x, labels)
    s = np.ndarray(shape=(0,1))
    counts = []
    xticks = []
    xlabels = []
    xmax = 0
    for i in range(no_components):
        idx = np.argwhere(labels==i)
        count = idx.shape[0]
        xticks.append(xmax+count/2)
        xlabels.append(r'$C_%i$' % i)
        xmax += count
        counts.append(count)
        s_i = np.sort(sil[idx], axis=0)
        s_i = np.flipud(s_i)
        s = np.concatenate((s,s_i))
    xvals = np.arange(xmax)
    plt.cla()
    plt.plot(xvals, s, 'k-')
    xval = 0
    ymin = min(np.min(s)*1.2,0)
    ymax = max(np.max(s)*1.2,0)
    for i in range(len(counts)):
        plt.text(xticks[i], ymax*(0.95-0.035*i), str(counts[i]), horizontalalignment='center')
        xval += counts[i]
        if i < len(counts)-1:
            plt.plot([xval,xval],[ymin,ymax],':r')
    plt.plot([0, xmax],[0, 0],':r')
    plt.axis([0,xmax,ymin,ymax])
    plt.xlabel('Samples by Cluster')
    plt.ylabel('Silhouette Coefficient')
    plt.xticks(xticks,xlabels)
    plt.tight_layout()
    suffix = '%i_clusters_silhouettes' % no_components
    fname = make_fname(method, model, dataset, suffix,'png')
    fpath = os.path.join(REPORT_HOME,fname)
    plt.savefig(fpath)


def project_data(x, mu, labels):
    """
    Project and high dimensional data onto a low dimensional
    hypersphere with dimensions = no clusters.
    @param x: xxx.
    @param mu: xxx.
    @param labels: xxx.
    @return pdata plabels: xxx.
    """

    x = x.toarray()
    cdata = []
    clabels = []

    for i in set(labels):
        x_i = x[np.argwhere(labels==i),:][:,0,:]
        cdata.append(x_i)
        clabels.append(np.array([i]*x_i.shape[0]))
    data = np.concatenate(cdata)
    pdata = np.dot(data, mu.T)
    plabels = np.concatenate(clabels)

    for i in range(pdata.shape[0]):
        norm = np.linalg.norm(pdata[i,:])
        if norm > 0:
            pdata[i,:] = pdata[i,:] / norm

    return pdata, plabels


