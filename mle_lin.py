import numpy as np
from scipy import optimize
from time import time
import scipy.linalg as spl
from utils import *

class LinBetaPLModel(object):
    '''
    reparametrize by pi = X*beta + b1
    minimize -loglikelihood wrt beta and b
    Constrained s.t. X*beta + b1 are all non-negative
    '''
    def __init__(self, endog, exog, X):
        '''
        n: number of items
        p: number of features
        M: number of rankings
        params: (beta, b), (p+1)*1
        :param endog: (dependent variable): {winner i}, M*1
        :param exog: (independent variable): {A}, M*k
        :param X: n*p, feature matrix
        '''
        self.endog = endog
        self.exog = exog
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = len(endog)
        self.X = X.astype(float)

    def fun_nonneg_constraint(self, params):
        return np.dot(self.X, params[:-1]) + params[-1] * np.ones((self.n,), dtype=float)

    def loglike(self, params):
        '''
        To be mininimized
        :param params: (beta, b), (p+1)*1
        '''
        ll = 0
        for ind_obs in range(self.M):
            i = self.endog[ind_obs].astype(int)  # winner
            A = self.exog[ind_obs].astype(int)  # alternatives
            sum_dot = 0
            for elm in A:
                sum_dot += np.dot(params[:-1], self.X[elm, :]) + params[-1]
            ll += np.log(np.dot(params[:-1], self.X[i, :] + epsilon) + params[-1]) - np.log(sum_dot + epsilon)
        return -ll

    def score(self, params):
        '''
        :param params: (beta, b), (p+1)*1
        '''
        grad = np.zeros(self.p+1, dtype=float)
        for ind_obs in range(self.M):
            i = self.endog[ind_obs].astype(int)  # winner
            A = self.exog[ind_obs].astype(int)  # alternatives
            sum_dot = 0
            for alt in A:
                sum_dot += np.dot(params[:-1], self.X[alt, :]) + params[-1]
            tmp = np.dot(params[:-1], self.X[i, :]) + params[-1]
            grad[:-1] += self.X[i, :] / tmp - np.sum(self.X[A, :], axis=0) / sum_dot
            grad[-1] += 1.0 / tmp - len(A) / sum_dot
        return -grad

    def fit(self, params=None, method='SLSQP', max_iter=10000):
        '''
        :param params: (beta, b), (p+1)*1
        :return: beta, b
        '''
        # define constraints, some margin for inequality constraints is required
        if method == 'SLSQP':
            # X * beta + b1 >= 0
            nonneg_constraint = {'type': 'ineq', 'fun': lambda params:
                    np.dot(self.X, params[:-1]) + params[-1] * np.ones((self.n,), dtype=float) - rtol}
            # sum(X * beta + b1) = 1
            # sum_constraint = {'type': 'eq', 'fun': lambda params:
            #        np.sum(np.dot(self.X, params[:-1]) + params[-1] * np.ones((self.n,), dtype=float)) - 1.0}
        elif method == 'trust-constr':
            # X * beta + b1 >= 0
            nonneg_constraint = optimize.NonlinearConstraint(self.fun_nonneg_constraint,
                                                rtol*np.ones((self.n,), dtype=float), np.ones((self.n,), dtype=float))
        else:
            # unconstrained
            nonneg_constraint = {}
        if params is None:
            # Initialization, start from a feasible point for all parameters
            # Initialization, start from a feasible point for all parameters
            (beta, b, _), _, _, _, _ = init_params(self.X, self.exog)
            params[:-1] = beta
            params[-1] = b
        start = time()
        res = optimize.minimize(self.loglike, params, method=method, jac=self.score, hess='2-point',
                                constraints=[nonneg_constraint], options={'maxiter': max_iter})
        end = time()
        return res.x[:-1], res.x[-1], (end - start)