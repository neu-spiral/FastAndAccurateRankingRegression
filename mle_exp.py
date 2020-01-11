import numpy as np
import statsmodels.base.model
from time import time
from utils import *

class ExpBetaPLModel(statsmodels.base.model.LikelihoodModel):
    '''
    reparametrize by pi = exp(X*beta)
    maximize loglikelihood(beta) wrt beta
    Unconstrained
    '''
    def __init__(self, endog, exog, X, **kwargs):
        '''
        n: number of items
        p: number of features
        M: number of rankings
        params: beta, p*1
        :param endog: (dependent variable): {winner i}, M*1
        :param exog: (independent variable): {A}, M*k
        :param X: n*p, feature matrix
        '''
        super(ExpBetaPLModel, self).__init__(endog, exog, **kwargs)
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = len(endog)
        self.X = X.astype(float)

    def loglike(self, params):
        ll = 0.0
        for ind_obs in range(self.M):
            i = self.endog[ind_obs].astype(int)  # winner
            A = self.exog[ind_obs].astype(int)  # alternatives
            second_term = 0
            for alt in A:
                second_term += np.exp(np.dot(params, self.X[alt, :]))
            ll += np.dot(params, self.X[i, :]) - np.log(second_term)
        return ll

    def score(self, params):
        grad = np.zeros(self.p, dtype=float)
        for ind_obs in range(self.M):
            i = self.endog[ind_obs].astype(int)  # winner
            A = self.exog[ind_obs].astype(int)  # alternatives
            second_term_num = np.zeros(self.p, dtype=float)
            second_term_denom = 0
            for alt in A:
                tmp = np.exp(np.dot(params, self.X[alt, :]))
                second_term_num += self.X[alt, :] * tmp
                second_term_denom += tmp
            grad += self.X[i, :] - second_term_num / second_term_denom
        return grad

    def hessian(self, params):
        hess = np.eye((self.p), dtype=float) * epsilon
        for ind_obs in range(self.M):
            A = self.exog[ind_obs].astype(int)  # alternatives
            first_term_num = np.zeros(self.p, dtype=float)
            second_term_num = np.zeros((self.p, self.p), dtype=float)
            denom = 0
            for alt in A:
                tmp = np.exp(np.dot(params, self.X[alt, :]))
                first_term_num += self.X[alt, :] * tmp
                second_term_num += np.outer(self.X[alt, :], self.X[alt, :]) * tmp
                denom += tmp
            hess += np.outer(first_term_num / denom, first_term_num / denom) - \
                    second_term_num / denom
        return hess

    def fit(self, params=None, max_iter=10000, **kwargs):
        if params is None:
            _, _, _, _, (exp_beta, _) = init_params(self.X, self.exog, mat_Pij=None)
            params = exp_beta
        start = time()
        res = super(ExpBetaPLModel, self).fit(start_params=params, method='newton', maxiter=max_iter, **kwargs)
        end = time()
        return res.params, (end - start)

