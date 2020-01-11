import numpy as np
import math
from time import time
import cvxpy as cp
import scipy.linalg as spl
import scipy.sparse.linalg as spsl
from scipy.stats import kendalltau
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix

# Global variables
epsilon = np.finfo(np.float).eps  ## machine precision
rtol = 1e-4  ## convergence tolerance
n_iter = 500

# Initialization, start from a feasible point for all parameters
def init_params(X, rankings,  mat_Pij, method_beta_b_init='QP'):
    '''
    n: number of items
    p: number of features
    :param rankings: (c_l, A_l): 1...M
    :param X: n*p, feature matrix
    :param mat_Pij = est_Pij(n, rankings)
    :param Q = sum for all pairs (i, j): [(P_ij x_j - P_ji x_i); (P_ij - P_ji)][(P_ij x_j - P_ji x_i); (P_ij - P_ji)]^T
    '''
    n = X.shape[0]
    if method_beta_b_init == 'LS':
        beta_init, b_init, time_beta_b_init = init_beta_b_ls(X=X, rankings=rankings)
    else:
        beta_init, b_init, time_beta_b_init = init_beta_b_convex_QP(X=X, rankings=rankings, mat_Pij=mat_Pij)
    ## weights are initialized uniformly as in all other methods
    start_pi = time()
    pi_init = 1.0 * np.ones(n, dtype=float) / n
    time_pi_init = time() - start_pi
    ## u is initialized
    start_u = time()
    u_init = np.zeros(n, dtype=float)
    time_u_init = time() - start_u
    ## theta newton is initialized
    start_theta = time()
    theta_init = np.zeros(n, dtype=float)
    time_theta_init = time() - start_theta
    ## beta newton exp beta is initialized
    exp_beta_init, time_exp_beta_init = init_exp_beta(X, rankings, mat_Pij)
    return (beta_init, b_init, time_beta_b_init), (pi_init, time_pi_init), (u_init, time_u_init), \
           (theta_init, time_theta_init), (exp_beta_init, time_exp_beta_init)

def init_beta_b_convex_QP(X, rankings, mat_Pij):
    '''
    n: number of items
    p: number of features
    :param rankings: (c_l, A_l): 1...M
    :param X: n*p, feature matrix
    :param mat_Pij = est_Pij(n, rankings)
    min._{beta,b} {beta,b}^T Q {beta,b}, s.t. Xbeta + b >=0 and sum(Xbeta + b)=1
    :return: beta, b, time
    '''
    p = X.shape[1]
    # Define variables
    beta = cp.Variable(p)
    b = cp.Variable(1)
    params = cp.vstack([beta, b])
    # Define objective
    Q = est_sum_dij_dijT(X, rankings, mat_Pij)
    objective = cp.quad_form(params, Q)
    # Define constraints
    constraints = [X * beta + b >= rtol,
                   cp.sum_entries(X * beta + b) == 1]
    start_beta_b = time()
    # Optimize
    prob = cp.Problem(cp.Minimize(objective), constraints=constraints)
    prob.solve(solver='SCS')
    time_beta_b_init = time() - start_beta_b
    if beta.value is None:
        constraints = [X * beta + b >= rtol]
        # If cannot be solved, reduce the accuracy requirement
        start_beta_b = time()
        # Optimize
        prob = cp.Problem(cp.Minimize(objective), constraints=constraints)
        prob.solve(solver='SCS', eps=1e-2)
        time_beta_b_init = time() - start_beta_b
    return np.squeeze(np.array(beta.value)), b.value, time_beta_b_init

def est_Pij(n, rankings):
    '''
    n: number of items
    p: number of features
    :param rankings: (c_l, A_l): 1...M
    :param X: n*p, feature matrix
    :return: for each pair (i, j), empirical estimate of the probability of i beating j
    sparse version for write once read multiple times:
    '''
    Pij = lil_matrix((n, n), dtype=float)
    for ranking in rankings:
        for i, winner in enumerate(ranking):
            for loser in ranking[i + 1:]:
                Pij[winner, loser] += 1
    copy_Pij = lil_matrix.copy(Pij)
    for i in range(n):
        for j in range(i, n):
            summation = copy_Pij[i, j] + copy_Pij[j, i]
            if summation:
                Pij[i, j] /= summation
                Pij[j, i] /= summation
    return csc_matrix(Pij)

def est_sum_dij_dijT(X, rankings, mat_Pij):
    '''
    n: number of items
    p: number of features
    :param rankings: (c_l, A_l): 1...M
    :param X: n*p, feature matrix
    :param mat_Pij = est_Pij(n, rankings)
    :return: sum for all pairs (i, j): [(P_ij x_j - P_ji x_i); (P_ij - P_ji)][(P_ij x_j - P_ji x_i); (P_ij - P_ji)]^T
    '''
    n = X.shape[0]
    p = X.shape[1]
    if mat_Pij is None:
        mat_Pij = est_Pij(n, rankings)
    sum_dij_dijT = np.zeros((p+1, p+1), dtype=float)
    items = np.unique(rankings)
    for i in items:
        for j in items:
            if i != j:
                d_ij = np.concatenate(((mat_Pij[i, j] * X[j, :] - mat_Pij[j, i] * X[i, :]),
                                       [mat_Pij[i, j] - mat_Pij[j, i]]))
                sum_dij_dijT += np.outer(d_ij, d_ij)
    return sum_dij_dijT

def init_beta_b_ls(X, rankings):
    '''
    least squares initialization for parameter vector beta and bias b
    n: number of items
    p: number of features
    :param rankings: (c_l, A_l): 1...M
    :param X: n*p, feature matrix
    :return: beta, b, time
    '''
    ## beta is initialized to minimize least squares on the labels, not the scores: (y_ij-(x_i-x_j)^T beta)^2
    sum_cross_corr = 0
    sum_auto_corr = 0
    for ranking in rankings:
        for i, winner in enumerate(ranking):
            for loser in ranking[i + 1:]:
                X_ij = X[winner, :] - X[loser, :]
                sum_cross_corr += X_ij
                sum_auto_corr += np.outer(X_ij, X_ij)
    start_beta_b = time()
    beta_init = np.linalg.solve(sum_auto_corr, sum_cross_corr)
    ## b is initialized so that X*beta + b is positive in the beginning
    x_beta_init = np.dot(X, beta_init)
    b_init = np.max(-1.0 * x_beta_init) + rtol
    ## Sum up to 1
    #sum_pi = np.sum(x_beta_init + b_init)
    #beta_init = beta_init / sum_pi
    #b_init = b_init / sum_pi
    time_beta_b_init = time() - start_beta_b
    return beta_init, b_init, time_beta_b_init

def init_exp_beta(X, rankings, mat_Pij):
    '''
    least squares initialization for beta in exponential parametrization
    (alternative: exp_beta_init = np.ones((p,), dtype=float) * epsilon)
    n: number of items
    p: number of features
    :param rankings: (c_l, A_l): 1...M
    :param X: n*p, feature matrix
    :param mat_Pij = est_Pij(n, rankings)
    :return: exp_beta, time
    '''
    n = X.shape[0]
    if mat_Pij is None:
        mat_Pij = est_Pij(n, rankings)
    sum_auto_corr = 0
    sum_cross_corr = 0
    items = np.unique(rankings)
    for i in items:
        for j in items:
            if i != j:
                X_ij = X[i, :] - X[j, :]
                sum_auto_corr += np.outer(X_ij, X_ij)
                if mat_Pij[i, j] > 0 and mat_Pij[j, i] > 0:
                    s_ij = mat_Pij[i, j] / mat_Pij[j, i]
                elif mat_Pij[i, j] == 0 and mat_Pij[j, i] == 0:
                    s_ij = 1
                elif mat_Pij[i, j] == 0 and mat_Pij[j, i] > 0:
                    s_ij = 1/len(rankings)
                elif mat_Pij[i, j] > 0 and mat_Pij[j, i] == 0:
                    s_ij = len(rankings)
                sum_cross_corr += np.log(s_ij) * X_ij
    start_exp_beta = time()
    exp_beta_init = spl.lstsq(sum_auto_corr, sum_cross_corr)[0]  # uses svd
    time_exp_beta_init = time() - start_exp_beta
    return exp_beta_init, time_exp_beta_init

def statdist(generator, method="power", v_init=None):
    """Compute the stationary distribution of a Markov chain, described by its infinitesimal generator matrix.
    Computing the stationary distribution can be done with one of the following methods:
    - `kernel`: directly computes the left null space (co-kernel) the generator
      matrix using its LU-decomposition. Alternatively: ns = spl.null_space(generator.T)
    - `eigenval`: finds the leading left eigenvector of an equivalent
      discrete-time MC using `scipy.sparse.linalg.eigs`.
    - `power`: finds the leading left eigenvector of an equivalent
      discrete-time MC using power iterations. v_init is the initial eigenvector.
    """
    n = generator.shape[0]
    if method == "kernel":
        # `lu` contains U on the upper triangle, including the diagonal.
        lu, piv = spl.lu_factor(generator.T, check_finite=False)
        # The last row contains 0's only.
        left = lu[:-1,:-1]
        right = -lu[:-1,-1]
        # Solves system `left * x = right`. Assumes that `left` is
        # upper-triangular (ignores lower triangle.)
        res = spl.solve_triangular(left, right, check_finite=False)
        res = np.append(res, 1.0)
        return (1.0 / res.sum()) * res
    if method == "eigenval":
        '''
        Arnoldi iteration has cubic convergence rate, but does not guarantee positive eigenvector
        '''
        if v_init is None:
            v_init = np.random.rand(n,)
        # mat = generator+eye is row stochastic, i.e. rows add up to 1.
        # Multiply by eps to make sure each entry is at least -1. Sum by 1 to make sure that each row sum is exactly 1.
        eps = 1.0 / np.max(np.abs(generator))
        mat = np.eye(n) + eps*generator
        A = mat.T
        # Find the leading left eigenvector, corresponding to eigenvalue 1
        _, vecs = spsl.eigs(A, k=1, v0=v_init)
        res = np.real(vecs[:,0])
        return (1.0 / res.sum()) * res
    if method == "power":
        '''
        Power iteration has linear convergence rate and slow for lambda2~lambda1. 
        But guarantees positive eigenvector, if started accordingly.
        '''
        if v_init is None:
            v = np.random.rand(n,)
        else:
            v = v_init
        # mat = generator+eye is row stochastic, i.e. rows add up to 1.
        # Multiply by eps to make sure each entry is at least -1. Sum by 1 to make sure that each row sum is exactly 1.
        eps = 1.0 / np.max(np.abs(generator))
        mat = np.eye(n) + eps * generator
        A = mat.T
        # Find the leading left eigenvector, corresponding to eigenvalue 1
        normAest = np.sqrt(np.linalg.norm(A, ord=1) * np.linalg.norm(A, ord=np.inf))
        v = v/np.linalg.norm(v)
        Av = np.dot(A,v)
        for ind_iter in range(n_iter):
            v = Av/np.linalg.norm(Av)
            Av = np.dot(A,v)
            lamda = np.dot(v.T, Av)
            r = Av-v*lamda
            normr = np.linalg.norm(r)
            if normr < rtol*normAest:
                #print('Power iteration converged in ' + str(ind_iter) + ' iterations.')
                break
        res = np.real(v)
        return (1.0 / res.sum()) * res
    else:
        raise RuntimeError("not (yet?) implemented")

def objective(weights, rankings):
    ll = 0
    # against log(0), add epsilon
    for ranking in rankings:
        sum_weights = sum(weights[x] for x in ranking)
        winner = ranking[0]
        ll += np.log(weights[winner] + epsilon) - np.log(sum_weights + epsilon)
    return ll

def check_global_balance_eqn(generator, state_vect):
    '''
    :param generator: Matrix containing Markov chain transitions
    We need to check the balance eqn. for each column, i.e. for all i for transitions lambda_ji
    :return: True if global balance equation is satisfied
    '''
    # Multiply by eps to make sure each entry is at least -1. Sum by 1 to make sure that each row sum is exactly 1.
    generator = 1.0 / np.max(np.abs(generator)) * generator
    for i, state_i in enumerate(state_vect):
        col_i = generator[:, i]
        row_i = generator[i, :]
        if np.abs((np.sum(row_i*state_i) - row_i[i]*state_i) - (np.dot(state_vect, col_i) - col_i[i]*state_i)) > epsilon:
            return False
    return True

def softmax(a):
    tmp = np.exp(a - np.max(a))
    return tmp / np.sum(tmp)

def top1_test_accuracy(est_scores, rankings_test):
    correct = 0
    for sample in rankings_test:
        k = len(sample)
        true_winner = sample[0]
        pl_probabilities = est_scores[sample]
        if not np.any(np.isnan(pl_probabilities)):
            # against equal weights
            if np.unique(pl_probabilities).shape[0] == 1:
                correct += 1 / k
            else:
                pred_winner = sample[np.argmax(pl_probabilities)]
                if true_winner == pred_winner:
                    correct += 1
    return 1.0 * correct / len(rankings_test)

def kendall_tau_test(est_scores, rankings_test):
    corr = 0
    for sample in rankings_test:
        pl_probabilities = est_scores[sample] / sum(est_scores[sample])
        if not np.any(np.isnan(pl_probabilities)):
            pred_ranking = sample[np.flip(np.argsort(pl_probabilities), axis=0)]
            corr += kendalltau(sample, pred_ranking)[0]  #between -1 and 1
    return 1.0 * corr / len(rankings_test)