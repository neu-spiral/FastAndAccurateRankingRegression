from __future__ import division

from time import time
import numpy as np
import statsmodels.base.model
from scipy.misc import logsumexp
from utils import *

def ilsr(n, rankings, weights=None):
    """Iterative Luce spectral ranking algorithm for partial ranking data. Remove the inner loop for top-1 ranking.
    For each ranking, there are (k-1) independent observations with winner=i and losers = {i+1,...,k}"""
    if weights is None:
        weights = 1.0 * np.ones(n, dtype=float) / n
    start = time()
    chain = np.zeros((n, n), dtype=float)
    for ranking in rankings:
        sum_weights = sum(weights[x] for x in ranking) + epsilon
        for i, winner in enumerate(ranking):
            val = 1.0 / sum_weights
            for loser in ranking[i + 1:]:
                chain[loser, winner] += val
            sum_weights -= weights[winner]
    # each row sums up to 0
    chain -= np.diag(chain.sum(axis=1))
    weights = statdist(chain, v_init=weights)
    end = time()
    return weights, (end - start)


def mm_iter(n, rankings, weights=None):
    """Hunter's minorization-maximization algorithm."""
    if weights is None:
        weights = 1.0 * np.ones(n, dtype=float) / n
    start = time()
    wts = np.zeros(n, dtype=float)
    denoms = np.zeros(n, dtype=float) + epsilon
    for ranking in rankings:
        # Each item ranked second to last or better receives 1.0.
        wts[list(ranking[:-1])] += 1.0
        sum_weights = sum(weights[x] for x in ranking) + epsilon
        for idx, i in enumerate(ranking[:-1]):
            val = 1.0 / sum_weights
            for s in ranking[idx:]:
                denoms[s] += val
            sum_weights -= weights[i]
    res = wts / denoms
    end = time()
    return res / res.sum(), (end - start)


class ConvexPLModel(statsmodels.base.model.LikelihoodModel):
    '''
    reparametrize by pi=e^theta
    maximize loglikelihood(theta) wrt theta
    Unconstrained
    '''
    def __init__(self, endog, exog, n, **kwargs):
        '''
        params: theta, n*1
        n: number of items
        :param endog: (dependent variable): {winner i}, M*1
        :param exog: (independent variable): {A}, M*k
        '''
        super(ConvexPLModel, self).__init__(endog, exog, **kwargs)
        self._n = n

    def loglike(self, params):
        params = np.append(params, 0.0)
        ll = 0.0
        for ids, x in zip(self.exog, self.endog):
            ll += params[x] - logsumexp(params[ids])
        return ll

    def score(self, params):
        params = np.append(params, 0.0)
        grad = np.zeros(self._n, dtype=float)
        for ids, x in zip(self.exog, self.endog):
            grad[ids] -= softmax(params[ids])
            grad[x] += 1
        return grad[:-1]

    def hessian(self, params):
        params = np.append(params, 0.0)
        hess = np.eye((self._n), dtype=float) * epsilon
        for ids in self.exog:
            vals = softmax(params[ids])
            hess[np.ix_(ids, ids)] += np.outer(vals, vals) - np.diag(vals)
        return hess[:-1,:-1]

    def fit(self, start_params=None, maxiter=10000, **kwargs):
        '''
        :param start_params: initial theta
        :return: final theta
        '''
        if start_params is None:
            # Reasonable starting values
            start_params = np.zeros(self._n - 1, dtype=float)
        start = time()
        res = super(ConvexPLModel, self).fit(start_params=start_params, maxiter=maxiter, method='newton', **kwargs)
        end = time()
        # Add the last parameter back, and zero-mean it for good measure.
        res.params = np.append(res.params, 0)
        res.params -= res.params.mean()
        return res.params, (end-start)




