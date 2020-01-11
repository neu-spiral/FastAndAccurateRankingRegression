import numpy as np
from time import time
import scipy.linalg as spl
import cvxpy as cp
from utils import *

class ADMM_log(object):
    def __init__(self, rankings, X, method_pi_tilde_init='prev'):
        '''
        n: number of items
        p: number of features
        M: number of rankings
        :param rankings: (c_l, A_l): 1...M
        :param X: n*p, feature matrix
        :param method_pi_tilde_init: for ilsr_feat, initialize with prev_weights or orthogonal projection
        '''
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = len(rankings)
        self.X = X.astype(float)
        self.X_ls = np.linalg.solve(np.dot(self.X.T, self.X), self.X.T)
        self.X_tilde = np.concatenate((self.X, np.ones((self.n,1))), axis=1)
        self.X_tilde_ls = np.linalg.solve(np.dot(self.X_tilde.T, self.X_tilde), self.X_tilde.T)
        self.rankings = rankings
        self.method_pi_tilde_init = method_pi_tilde_init

    def fit_log(self, rho, weights=None, beta=None, u=None, gamma=1):
        '''
        :param rho: penalty parameter
        :param beta: parameter vector at each iteration, px1
        :param weights: scores at each iteration, nx1
        :param u: scaled dual variable at each iteration, nx1
        :param gamma: scaling on the dual variable update
        '''
        if beta is None:
            _, _, (u, _), _, (beta, _) = init_params(self.X, self.rankings, mat_Pij=None)
            weights = softmax(np.dot(self.X, beta))
        start = time()
        ## beta update
        # beta = spl.lstsq(self.X, np.log(weights) - u)[0]  # uses svd
        beta = np.dot(self.X_ls, np.log(weights + epsilon) - u)
        x_beta = np.dot(self.X, beta)
        ## pi update
        weights = self.ilsrx_log(rho=rho, weights=weights, x_beta=x_beta, u=u)
        ## dual update
        u += gamma * (x_beta - np.log(weights + epsilon))
        end = time()
        return weights, beta, u, (end - start)

    def ilsrx_log(self, rho, weights, x_beta, u):
        """modified spectral ranking algorithm for partial ranking data. Remove the inner loop for top-1 ranking.
        n: number of items
        rho: penalty parameter
        sigmas = rho * (log(weights) - Xbeta - u)/weights is the additional term compared to ILSR
        """
        if self.method_pi_tilde_init == 'OP':
            sigmas = rho * (np.log(weights + epsilon) - x_beta - u)/(weights + epsilon)
            weights = self.init_ilsr_feat_convex_QP(weights, sigmas)
        ilsr_conv = False
        iter = 0
        while not ilsr_conv:
            sigmas = rho * (np.log(weights + epsilon) - x_beta - u)/(weights + epsilon)
            pi_sigmas = weights * sigmas
            #######################
            # print('Log ADMM 0-mean', np.sum(pi_sigmas))
            # indices of states for which sigmas < 0
            ind_minus = np.where(sigmas < 0)[0]
            # indices of states for which sigmas >= 0
            ind_plus = np.where(sigmas >= 0)[0]
            # sum of pi_sigmas over states for which sigmas >= 0
            scaled_sigmas_plus = sigmas[ind_plus] / np.sum(pi_sigmas[ind_minus])
            # fill up the transition matrix
            chain = np.zeros((self.n, self.n), dtype=float)
            # increase the outgoing rate from ind_plus to ind_minus
            for ind_minus_cur in ind_minus:
                chain[ind_plus, ind_minus_cur] = pi_sigmas[ind_minus_cur] * scaled_sigmas_plus
            for ranking in self.rankings:
                sum_weights = sum(weights[x] for x in ranking) + epsilon
                for i, winner in enumerate(ranking):
                    val = 1.0 / sum_weights
                    for loser in ranking[i + 1:]:
                        chain[loser, winner] += val
                    sum_weights -= weights[winner]
            # each row sums up to 0
            chain -= np.diag(chain.sum(axis=1))
            weights_prev = weights
            weights = statdist(chain, v_init=weights)
            # Check convergence
            iter += 1
            ilsr_conv = np.linalg.norm(weights_prev - weights) < rtol * np.linalg.norm(weights) or iter >= n_iter
        # print('Log ADMM balance', check_global_balance_eqn(chain, weights))
        return weights

    def init_ilsr_feat_convex_QP(self, weights_prev, sigmas):
        """
        sigmas is the additional term compared to ILSR
        min._{pi} ||pi-pi_{t-1}||^2, s.t. pi >=0 and sum(pi)=1 and sum(pi*sigma)=0
        :return initial weights for ilsr_feat which satisfy the mean-zero condition for MC
        """
        # Define variables
        weights = cp.Variable(self.n)
        # Define objective
        objective = cp.square(cp.norm(weights - weights_prev))
        # Define constraints
        constraints = [weights >= rtol,
                       cp.sum_entries(weights) == 1,
                       weights.T * sigmas == 0]
        # Optimize
        prob = cp.Problem(cp.Minimize(objective), constraints=constraints)
        prob.solve(solver='SCS')
        return np.squeeze(np.array(weights.value))
