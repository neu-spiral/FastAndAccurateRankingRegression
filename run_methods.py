from utils import *
from mle_lin import *
from mle_exp import *
from only_scores import *
from admm_lin import *
from admm_log import *
import pickle
import csv
from math import ceil
import argparse
from os.path import exists
from scipy.sparse import save_npz, load_npz

def init_all_methods_synthetic(dir, n, p, k, d, rand_iter, test_fold):
    '''
    Load synthetic data and initialize all methods
    :param dir: data directory
    :param n: number of items
    :param p: dimension
    :param k: size of partial ranking
    :param d: number of full rankings
    :param rand_iter: index of the random generation
    '''
    save_name = str(rand_iter) + '_' + str(test_fold) + '_n_' + str(n) + '_p_' + str(p) + '_k_' + str(k) + '_d_' + str(d)
    # train data
    rankings_train = np.load('../data/' + dir + 'data/' + save_name + '_train.npy')
    X = np.load('../data/' + dir + 'data/' + save_name + '_features.npy')
    # test data
    true_scores = np.load('../data/' + dir + 'data/' + save_name + '_scores.npy')
    true_beta = np.load('../data/' + dir + 'data/' + save_name + '_parameters.npy')
    rankings_test = np.load('../data/' + dir + 'data/' + save_name + '_test.npy')
    mat_Pij = load_npz('../data/' + dir + 'data/' + save_name + '_mat_Pij.npz')
    endog = rankings_train[:, 0]  # (dependent variable): {winner i}, M*1
    exog = rankings_train  # (independent variable): {A}, M*k
    ###############################
    # Initialization, start from a feasible point for all parameters
    (beta_init, b_init, time_beta_b_init), (pi_init, time_pi_init), (u_init, time_u_init), \
                        (theta_init, time_theta_init), (exp_beta_init, time_exp_beta_init) = \
                        init_params(X, rankings_train, mat_Pij, method_beta_b_init='QP')
    # Log all results
    log_dict = dict()
    # lsr parameters
    log_dict['lsr_conv'] = False
    log_dict['pi_lsr'] = np.copy(pi_init)
    log_dict['time_lsr'] = [time_pi_init]
    log_dict['diff_pi_lsr'] = [np.linalg.norm(true_scores - log_dict['pi_lsr'])]
    log_dict['obj_lsr'] = [objective(log_dict['pi_lsr'], rankings_train)]
    log_dict['test_acc_lsr'] = [top1_test_accuracy(log_dict['pi_lsr'], rankings_test)]
    log_dict['test_kendall_lsr'] = [kendall_tau_test(log_dict['pi_lsr'], rankings_test)]
    log_dict['iter_lsr'] = 0
    # mm parameters
    log_dict['mm_conv'] = False
    log_dict['pi_mm'] = np.copy(pi_init)
    log_dict['time_mm'] = [time_pi_init]
    log_dict['diff_pi_mm'] = [np.linalg.norm(true_scores - log_dict['pi_mm'])]
    log_dict['obj_mm'] = [objective(log_dict['pi_mm'], rankings_train)]
    log_dict['test_acc_mm'] = [top1_test_accuracy(log_dict['pi_mm'], rankings_test)]
    log_dict['test_kendall_mm'] = [kendall_tau_test(log_dict['pi_mm'], rankings_test)]
    log_dict['iter_mm'] = 0
    # linear admm parameters
    log_dict['lin_admm'] = ADMM_lin(rankings_train, X, method_pi_tilde_init='prev')
    log_dict['lin_admm_conv'] = False
    log_dict['beta_lin_admm'] = np.copy(beta_init)
    log_dict['b_lin_admm'] = b_init
    log_dict['pi_lin_admm'] = np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm']
    log_dict['u_lin_admm'] = np.copy(u_init)
    log_dict['time_lin_admm'] = [time_beta_b_init + time_u_init]
    log_dict['diff_pi_lin_admm'] = [np.linalg.norm(true_scores - log_dict['pi_lin_admm'])]
    log_dict['prim_feas_lin_admm'] = [np.linalg.norm(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'] - log_dict['pi_lin_admm'])]
    log_dict['dual_feas_lin_admm'] = [np.linalg.norm(np.dot(log_dict['lin_admm'].X_tilde.T, log_dict['pi_lin_admm']))]
    log_dict['obj_lin_admm'] = [objective(log_dict['pi_lin_admm'], rankings_train)]
    log_dict['test_acc_lin_admm'] = [top1_test_accuracy(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'], rankings_test)]
    log_dict['test_kendall_lin_admm'] = [kendall_tau_test(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'], rankings_test)]
    log_dict['iter_lin_admm'] = 0
    # log admm parameters
    log_dict['log_admm'] = ADMM_log(rankings_train, X, method_pi_tilde_init='prev')
    log_dict['log_admm_conv'] = False
    log_dict['beta_log_admm'] = np.copy(exp_beta_init)
    log_dict['pi_log_admm'] = softmax(np.dot(X, log_dict['beta_log_admm']))
    log_dict['u_log_admm'] = np.copy(u_init)
    log_dict['time_log_admm'] = [time_exp_beta_init + time_u_init]
    log_dict['diff_pi_log_admm'] = [np.linalg.norm(true_scores - log_dict['pi_log_admm'])]
    log_dict['diff_beta_log_admm'] = [np.linalg.norm(true_beta - log_dict['beta_log_admm'])]
    log_dict['prim_feas_log_admm'] = [np.linalg.norm(np.dot(X, log_dict['beta_log_admm']) - np.log(log_dict['pi_log_admm'] + epsilon))]
    log_dict['dual_feas_log_admm'] = [np.linalg.norm(np.dot(log_dict['log_admm'].X.T, np.log(log_dict['pi_log_admm'] + epsilon)))]
    log_dict['obj_log_admm'] = [objective(log_dict['pi_log_admm'], rankings_train)]
    log_dict['test_acc_log_admm'] = [top1_test_accuracy(softmax(np.dot(X, log_dict['beta_log_admm'])), rankings_test)]
    log_dict['test_kendall_log_admm'] = [kendall_tau_test(softmax(np.dot(X, log_dict['beta_log_admm'])), rankings_test)]
    log_dict['iter_log_admm'] = 0
    # convex newton parameters
    log_dict['theta_newton_conv'] = False
    log_dict['mle_theta'] = ConvexPLModel(endog=endog, exog=exog, n=n)
    log_dict['theta_newton'] = np.copy(theta_init)
    log_dict['pi_newton'] = softmax(log_dict['theta_newton'])
    log_dict['time_newton_theta'] = [time_theta_init]
    log_dict['diff_pi_newton_theta'] = [np.linalg.norm(true_scores - log_dict['pi_newton'])]
    log_dict['obj_newton_theta'] = [objective(log_dict['pi_newton'], rankings_train)]
    log_dict['test_acc_newton_theta'] = [top1_test_accuracy(log_dict['pi_newton'], rankings_test)]
    log_dict['test_kendall_newton_theta'] = [kendall_tau_test(log_dict['pi_newton'], rankings_test)]
    log_dict['iter_newton_theta'] = 0
    # unconstrained newton beta parameters
    log_dict['beta_newton_exp_beta_conv'] = False
    log_dict['mle_exp_beta'] = ExpBetaPLModel(endog=endog, exog=exog, X=X)
    log_dict['beta_newton_exp_beta'] = np.copy(exp_beta_init)
    log_dict['pi_newton_exp_beta'] = softmax(np.dot(X, log_dict['beta_newton_exp_beta']))
    log_dict['time_newton_exp_beta'] = [time_exp_beta_init]
    log_dict['diff_pi_newton_exp_beta'] = [np.linalg.norm(true_scores - log_dict['pi_newton_exp_beta'])]
    log_dict['diff_beta_newton_exp_beta'] = [np.linalg.norm(true_beta - log_dict['beta_newton_exp_beta'])]
    log_dict['obj_newton_exp_beta'] = [objective(log_dict['pi_newton_exp_beta'], rankings_train)]
    log_dict['test_acc_newton_exp_beta'] = [top1_test_accuracy(log_dict['pi_newton_exp_beta'], rankings_test)]
    log_dict['test_kendall_newton_exp_beta'] = [kendall_tau_test(log_dict['pi_newton_exp_beta'], rankings_test)]
    log_dict['iter_newton_exp_beta'] = 0
    # constrained slsqp beta parameters, normalize scores for comparison
    log_dict['slsqp_conv'] = False
    log_dict['mle_beta'] = LinBetaPLModel(endog=endog, exog=exog, X=X)
    log_dict['beta_slsqp'] = np.copy(beta_init)
    log_dict['b_slsqp'] = b_init
    log_dict['pi_slsqp'] = (np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp']) / np.sum(np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp'])
    log_dict['time_slsqp'] = [time_beta_b_init]
    log_dict['diff_pi_slsqp'] = [np.linalg.norm(true_scores - log_dict['pi_slsqp'])]
    log_dict['obj_slsqp'] = [objective(log_dict['pi_slsqp'], rankings_train)]
    log_dict['test_acc_slsqp'] = [top1_test_accuracy(log_dict['pi_slsqp'], rankings_test)]
    log_dict['test_kendall_slsqp'] = [kendall_tau_test(log_dict['pi_slsqp'], rankings_test)]
    log_dict['iter_slsqp'] = 0
    return log_dict, save_name, rankings_train, X, true_scores, true_beta, rankings_test, mat_Pij, endog, exog


def run_save_all_methods_synthetic(tasks, dir, n, p, k, d, rand_iter, test_fold, rho=1):
    '''
    Run all methods and save all logged results
    :param tasks: list of algorithms to be run
    :param dir: data directory
    :param n: number of items
    :param p: dimension
    :param k: size of partial ranking
    :param d: number of full rankings
    :param rand_iter: index of the random generation
    :param test_fold: index of test fold
    :param rho: penalty parameter of ADMM
    '''
    log_dict, save_name, rankings_train, X, true_scores, true_beta, rankings_test, mat_Pij, endog, exog = \
                        init_all_methods_synthetic(dir, n, p, k, d, rand_iter, test_fold)
    for iter in range(n_iter):
        # lsr update
        if not log_dict['lsr_conv'] and 'lsr' in tasks:
            log_dict['pi_lsr_prev'] = log_dict['pi_lsr']
            log_dict['pi_lsr'], time_lsr_iter = ilsr(n, rankings_train, weights=log_dict['pi_lsr'])
            log_dict['time_lsr'].append(time_lsr_iter)
            log_dict['diff_pi_lsr'].append(np.linalg.norm(true_scores - log_dict['pi_lsr']))
            log_dict['obj_lsr'].append(objective(log_dict['pi_lsr'], rankings_train))
            log_dict['test_acc_lsr'].append(top1_test_accuracy(log_dict['pi_lsr'], rankings_test))
            log_dict['test_kendall_lsr'].append(kendall_tau_test(log_dict['pi_lsr'], rankings_test))
            log_dict['iter_lsr'] += 1
            log_dict['lsr_conv'] = np.linalg.norm(log_dict['pi_lsr_prev'] - log_dict['pi_lsr']) < rtol * np.linalg.norm(log_dict['pi_lsr'])
        # mm update
        if not log_dict['mm_conv'] and 'mm' in tasks:
            log_dict['pi_mm_prev'] = log_dict['pi_mm']
            log_dict['pi_mm'], time_mm_iter = mm_iter(n, rankings_train, weights=log_dict['pi_mm'])
            if np.any(np.isnan(log_dict['pi_mm'])):
                log_dict['mm_conv'] = True
            else:
                log_dict['time_mm'].append(time_mm_iter)
                log_dict['diff_pi_mm'].append(np.linalg.norm(true_scores - log_dict['pi_mm']))
                log_dict['obj_mm'].append(objective(log_dict['pi_mm'], rankings_train))
                log_dict['test_acc_mm'].append(top1_test_accuracy(log_dict['pi_mm'], rankings_test))
                log_dict['test_kendall_mm'].append(kendall_tau_test(log_dict['pi_mm'], rankings_test))
                log_dict['iter_mm'] += 1
                log_dict['mm_conv'] = np.linalg.norm(log_dict['pi_mm_prev'] - log_dict['pi_mm']) < rtol * np.linalg.norm(log_dict['pi_mm'])
        # lin_admm update
        if not log_dict['lin_admm_conv'] and 'lin_admm' in tasks:
            log_dict['pi_lin_admm_prev'] = log_dict['pi_lin_admm']
            log_dict['beta_lin_admm_prev'] = log_dict['beta_lin_admm']
            log_dict['tilde_pi_lin_admm_prev'] = (np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm']) / np.sum(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'])
            log_dict['pi_lin_admm'], log_dict['beta_lin_admm'], log_dict['b_lin_admm'], log_dict['u_lin_admm'], time_lin_admm_iter = \
                log_dict['lin_admm'].fit_lin(rho, weights=log_dict['pi_lin_admm'], beta=log_dict['beta_lin_admm'], b=log_dict['b_lin_admm'], u=log_dict['u_lin_admm'])
            # scores predicted by beta
            log_dict['tilde_pi_lin_admm'] = (np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm']) / np.sum(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'])
            log_dict['time_lin_admm'].append(time_lin_admm_iter)
            log_dict['diff_pi_lin_admm'].append(np.linalg.norm(true_scores - log_dict['tilde_pi_lin_admm']))
            log_dict['prim_feas_lin_admm'].append(np.linalg.norm(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'] - log_dict['pi_lin_admm']))
            log_dict['dual_feas_lin_admm'].append(np.linalg.norm(np.dot(log_dict['lin_admm'].X_tilde.T, log_dict['pi_lin_admm_prev'] - log_dict['pi_lin_admm'])))
            log_dict['obj_lin_admm'].append(objective(log_dict['pi_lin_admm'], rankings_train))
            log_dict['test_acc_lin_admm'].append(top1_test_accuracy(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'], rankings_test))
            log_dict['test_kendall_lin_admm'].append(kendall_tau_test(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'], rankings_test))
            log_dict['iter_lin_admm'] += 1
            #log_dict['lin_admm_conv'] = log_dict['prim_feas_lin_admm'][-1] < rtol * np.max(
            #        [np.linalg.norm(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm']),
            #        np.linalg.norm(log_dict['pi_lin_admm'])]) and log_dict['dual_feas_lin_admm'][-1] < \
            #        rtol * np.linalg.norm(np.dot(log_dict['lin_admm'].X_tilde.T, log_dict['u_lin_admm']))
            log_dict['lin_admm_conv'] = np.linalg.norm(log_dict['pi_lin_admm_prev'] - log_dict['pi_lin_admm']) < rtol * np.linalg.norm(log_dict['pi_lin_admm']) \
                    and np.linalg.norm(log_dict['tilde_pi_lin_admm_prev'] - log_dict['tilde_pi_lin_admm']) < rtol * np.linalg.norm(log_dict['tilde_pi_lin_admm'])
        # log_admm update
        if not log_dict['log_admm_conv'] and 'log_admm' in tasks:
            log_dict['pi_log_admm_prev'] = log_dict['pi_log_admm']
            log_dict['beta_log_admm_prev'] = log_dict['beta_log_admm']
            log_dict['tilde_pi_log_admm_prev'] = softmax(np.dot(X, log_dict['beta_log_admm']))
            log_dict['pi_log_admm'], log_dict['beta_log_admm'], log_dict['u_log_admm'], time_log_admm_iter = \
                log_dict['log_admm'].fit_log(rho, weights=log_dict['pi_log_admm'], beta=log_dict['beta_log_admm'], u=log_dict['u_log_admm'])
            # scores predicted by beta
            log_dict['tilde_pi_log_admm'] = softmax(np.dot(X, log_dict['beta_log_admm']))
            log_dict['time_log_admm'].append(time_log_admm_iter)
            log_dict['diff_pi_log_admm'].append(np.linalg.norm(true_scores - log_dict['tilde_pi_log_admm']))
            log_dict['diff_beta_log_admm'].append(np.linalg.norm(true_beta - log_dict['beta_log_admm']))
            log_dict['prim_feas_log_admm'].append(np.linalg.norm(np.dot(X, log_dict['beta_log_admm']) - np.log(log_dict['pi_log_admm'] + epsilon)))
            log_dict['dual_feas_log_admm'].append(np.linalg.norm(
                np.dot(log_dict['log_admm'].X.T, np.log(log_dict['pi_log_admm_prev'] + epsilon) - np.log(log_dict['pi_log_admm'] + epsilon))))
            log_dict['obj_log_admm'].append(objective(log_dict['pi_log_admm'], rankings_train))
            log_dict['test_acc_log_admm'].append(top1_test_accuracy(softmax(np.dot(X, log_dict['beta_log_admm'])), rankings_test))
            log_dict['test_kendall_log_admm'].append(kendall_tau_test(softmax(np.dot(X, log_dict['beta_log_admm'])), rankings_test))
            log_dict['iter_log_admm'] += 1
            #log_dict['log_admm_conv'] = log_dict['prim_feas_log_admm'][-1] < rtol * np.max([np.linalg.norm(np.dot(X, log_dict['beta_log_admm'])),
            #        np.linalg.norm(np.log(log_dict['pi_log_admm'] + epsilon))]) and log_dict['dual_feas_log_admm'][-1] < \
            #        rtol * np.linalg.norm(np.dot(log_dict['log_admm'].X.T, log_dict['u_log_admm']))
            log_dict['log_admm_conv'] = np.linalg.norm(log_dict['pi_log_admm_prev'] - log_dict['pi_log_admm']) < rtol * np.linalg.norm(log_dict['pi_log_admm']) \
                    and np.linalg.norm(log_dict['tilde_pi_log_admm_prev'] - log_dict['tilde_pi_log_admm']) < rtol * np.linalg.norm(log_dict['tilde_pi_log_admm'])
        # convex newton update
        if not log_dict['theta_newton_conv'] and 'theta_newton' in tasks:
            try:
                log_dict['theta_newton_prev'] = log_dict['theta_newton']
                log_dict['pi_newton_prev'] = softmax(log_dict['theta_newton'])
                log_dict['theta_newton'], time_newton_theta_iter = log_dict['mle_theta'].fit(start_params=log_dict['theta_newton'][:-1], maxiter=1)
                log_dict['pi_newton'] = softmax(log_dict['theta_newton'])
                log_dict['time_newton_theta'].append(time_newton_theta_iter)
                log_dict['diff_pi_newton_theta'].append(np.linalg.norm(true_scores - log_dict['pi_newton']))
                log_dict['obj_newton_theta'].append(objective(log_dict['pi_newton'], rankings_train))
                log_dict['test_acc_newton_theta'].append(top1_test_accuracy(log_dict['pi_newton'], rankings_test))
                log_dict['test_kendall_newton_theta'].append(kendall_tau_test(log_dict['pi_newton'], rankings_test))
                log_dict['iter_newton_theta'] += 1
                log_dict['theta_newton_conv'] = np.linalg.norm(log_dict['pi_newton_prev'] - log_dict['pi_newton']) < rtol * np.linalg.norm(log_dict['pi_newton'])
            except np.linalg.LinAlgError:
                print('Convex Newton diverged')
        # unconstrained newton beta parameters
        if not log_dict['beta_newton_exp_beta_conv'] and 'beta_newton_exp_beta' in tasks:
            try:
                log_dict['beta_newton_exp_beta_prev'] = log_dict['beta_newton_exp_beta']
                log_dict['pi_newton_exp_beta_prev'] = softmax(np.dot(X, log_dict['beta_newton_exp_beta']))
                log_dict['beta_newton_exp_beta'], time_newton_exp_beta_iter = log_dict['mle_exp_beta'].fit(params=log_dict['beta_newton_exp_beta'], max_iter=1)
                if np.any(np.isnan(log_dict['beta_newton_exp_beta'])):
                    log_dict['beta_newton_exp_beta_conv'] = True
                else:
                    log_dict['pi_newton_exp_beta'] = softmax(np.dot(X, log_dict['beta_newton_exp_beta']))
                    log_dict['time_newton_exp_beta'].append(time_newton_exp_beta_iter)
                    log_dict['diff_pi_newton_exp_beta'].append(np.linalg.norm(true_scores - log_dict['pi_newton_exp_beta']))
                    log_dict['diff_beta_newton_exp_beta'].append(np.linalg.norm(true_beta - log_dict['beta_newton_exp_beta']))
                    log_dict['obj_newton_exp_beta'].append(objective(log_dict['pi_newton_exp_beta'], rankings_train))
                    log_dict['test_acc_newton_exp_beta'].append(top1_test_accuracy(log_dict['pi_newton_exp_beta'], rankings_test))
                    log_dict['test_kendall_newton_exp_beta'].append(kendall_tau_test(log_dict['pi_newton_exp_beta'], rankings_test))
                    log_dict['iter_newton_exp_beta'] += 1
                    log_dict['beta_newton_exp_beta_conv'] = np.linalg.norm(log_dict['pi_newton_exp_beta_prev'] - log_dict['pi_newton_exp_beta']) < rtol * np.linalg.norm(log_dict['pi_newton_exp_beta'])
            except np.linalg.LinAlgError:
                print('Newton on Beta diverged')
        # slsqp update, normalize scores for comparison
        if not log_dict['slsqp_conv'] and 'slsqp' in tasks:
            log_dict['beta_slsqp_prev'] = log_dict['beta_slsqp']
            log_dict['b_slsqp_prev'] = log_dict['b_slsqp']
            log_dict['pi_slsqp_prev'] = (np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp']) / np.sum(np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp'])
            log_dict['beta_slsqp'], log_dict['b_slsqp'], time_slsqp_iter = log_dict['mle_beta'].fit(params=np.concatenate((log_dict['beta_slsqp'],
                    [log_dict['b_slsqp']])), method='SLSQP', max_iter=1)
            log_dict['pi_slsqp'] = (np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp']) / np.sum(np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp'])
            log_dict['time_slsqp'].append(time_slsqp_iter)
            log_dict['diff_pi_slsqp'].append(np.linalg.norm(true_scores - log_dict['pi_slsqp']))
            log_dict['obj_slsqp'].append(objective(log_dict['pi_slsqp'], rankings_train))
            log_dict['test_acc_slsqp'].append(top1_test_accuracy(log_dict['pi_slsqp'], rankings_test))
            log_dict['test_kendall_slsqp'].append(kendall_tau_test(log_dict['pi_slsqp'], rankings_test))
            log_dict['iter_slsqp'] += 1
            log_dict['slsqp_conv'] = np.linalg.norm(log_dict['pi_slsqp_prev'] - log_dict['pi_slsqp']) < rtol * np.linalg.norm(log_dict['pi_slsqp'])
        # stop if all converged
        if log_dict['lsr_conv'] and log_dict['mm_conv'] and log_dict['lin_admm_conv'] and log_dict['log_admm_conv'] and log_dict['theta_newton_conv'] \
                    and log_dict['beta_newton_exp_beta_conv'] and log_dict['slsqp_conv']:
            break
    # Correct time scale
    log_dict['time_cont_lsr'] = [sum(log_dict['time_lsr'][:ind + 1]) for ind in range(len(log_dict['time_lsr']))]
    log_dict['time_cont_mm'] = [sum(log_dict['time_mm'][:ind + 1]) for ind in range(len(log_dict['time_mm']))]
    log_dict['time_cont_lin_admm'] = [sum(log_dict['time_lin_admm'][:ind + 1]) for ind in range(len(log_dict['time_lin_admm']))]
    log_dict['time_cont_log_admm'] = [sum(log_dict['time_log_admm'][:ind + 1]) for ind in range(len(log_dict['time_log_admm']))]
    log_dict['time_cont_newton_theta'] = [sum(log_dict['time_newton_theta'][:ind + 1]) for ind in range(len(log_dict['time_newton_theta']))]
    log_dict['time_cont_newton_exp_beta'] = [sum(log_dict['time_newton_exp_beta'][:ind + 1]) for ind in range(len(log_dict['time_newton_exp_beta']))]
    log_dict['time_cont_slsqp'] = [sum(log_dict['time_slsqp'][:ind + 1]) for ind in range(len(log_dict['time_slsqp']))]
    # Save results as a csv file
    save_name = save_name + '_rho_' + str(rho)
    with open('../results/' + dir + 'fig/' + '_logs_' + save_name + '.pickle', "wb") as pickle_out:
        pickle.dump(log_dict, pickle_out)
        pickle_out.close()
    return log_dict, save_name


def init_all_methods_real_data(dir, d, test_fold):
    save_name = str(test_fold) + '_d_' + str(d)
    # load data
    rankings_train = np.load('../data/' + dir + 'data/' + save_name + '_train.npy')
    rankings_test = np.load('../data/' + dir + 'data/' + save_name + '_test.npy')
    X = np.load('../data/' + dir + 'data/' + save_name + '_features.npy').astype(float)
    mat_Pij = load_npz('../data/' + dir + 'data/' + save_name + '_mat_Pij.npz')
    endog = rankings_train[:, 0]
    exog = rankings_train
    n = X.shape[0]  # number of items
    ###############################
    # Initialization, start from a feasible point for all parameters
    (beta_init, b_init, time_beta_b_init), (pi_init, time_pi_init), (u_init, time_u_init), \
            (theta_init, time_theta_init), (exp_beta_init, time_exp_beta_init) = \
                init_params(X, rankings_train, mat_Pij, method_beta_b_init='QP')
    # Log all results
    log_dict = dict()
    # lsr parameters
    log_dict['lsr_conv'] = False
    log_dict['pi_lsr'] = np.copy(pi_init)
    log_dict['time_lsr'] = [time_pi_init]
    log_dict['diff_pi_lsr'] = [np.linalg.norm(log_dict['pi_lsr'])]
    log_dict['obj_lsr'] = [objective(log_dict['pi_lsr'], rankings_train)]
    log_dict['test_acc_lsr'] = [top1_test_accuracy(log_dict['pi_lsr'], rankings_test)]
    log_dict['test_kendall_lsr'] = [kendall_tau_test(log_dict['pi_lsr'], rankings_test)]
    log_dict['iter_lsr'] = 0
    # mm parameters
    log_dict['mm_conv'] = False
    log_dict['pi_mm'] = np.copy(pi_init)
    log_dict['time_mm'] = [time_pi_init]
    log_dict['diff_pi_mm'] = [np.linalg.norm(log_dict['pi_mm'])]
    log_dict['obj_mm'] = [objective(log_dict['pi_mm'], rankings_train)]
    log_dict['test_acc_mm'] = [top1_test_accuracy(log_dict['pi_mm'], rankings_test)]
    log_dict['test_kendall_mm'] = [kendall_tau_test(log_dict['pi_mm'], rankings_test)]
    log_dict['iter_mm'] = 0
    # linear admm parameters
    log_dict['lin_admm'] = ADMM_lin(rankings_train, X, method_pi_tilde_init='prev')
    log_dict['lin_admm_conv'] = False
    log_dict['beta_lin_admm'] = np.copy(beta_init)
    log_dict['b_lin_admm'] = b_init
    log_dict['pi_lin_admm'] = np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm']
    log_dict['u_lin_admm'] = np.copy(u_init)
    log_dict['time_lin_admm'] = [time_beta_b_init + time_u_init]
    log_dict['diff_pi_lin_admm'] = [np.linalg.norm(log_dict['pi_lin_admm'])]
    log_dict['diff_beta_lin_admm'] = [np.linalg.norm(log_dict['beta_lin_admm'])]
    log_dict['prim_feas_lin_admm'] = [np.linalg.norm(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'] - log_dict['pi_lin_admm'])]
    log_dict['dual_feas_lin_admm'] = [np.linalg.norm(np.dot(log_dict['lin_admm'].X_tilde.T, log_dict['pi_lin_admm']))]
    log_dict['obj_lin_admm'] = [objective(log_dict['pi_lin_admm'], rankings_train)]
    log_dict['test_acc_lin_admm'] = [top1_test_accuracy(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'], rankings_test)]
    log_dict['test_kendall_lin_admm'] = [kendall_tau_test(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'], rankings_test)]
    log_dict['iter_lin_admm'] = 0
    # log admm parameters
    log_dict['log_admm'] = ADMM_log(rankings_train, X, method_pi_tilde_init='prev')
    log_dict['log_admm_conv'] = False
    log_dict['beta_log_admm'] = np.copy(exp_beta_init)
    log_dict['pi_log_admm'] = softmax(np.dot(X, log_dict['beta_log_admm']))
    log_dict['u_log_admm'] = np.copy(u_init)
    log_dict['time_log_admm'] = [time_exp_beta_init + time_u_init]
    log_dict['diff_pi_log_admm'] = [np.linalg.norm(log_dict['pi_log_admm'])]
    log_dict['diff_beta_log_admm'] = [np.linalg.norm(log_dict['beta_log_admm'])]
    log_dict['prim_feas_log_admm'] = [np.linalg.norm(np.dot(X, log_dict['beta_log_admm']) - np.log(log_dict['pi_log_admm'] + epsilon))]
    log_dict['dual_feas_log_admm'] = [np.linalg.norm(np.dot(log_dict['log_admm'].X.T, np.log(log_dict['pi_log_admm'] + epsilon)))]
    log_dict['obj_log_admm'] = [objective(log_dict['pi_log_admm'], rankings_train)]
    log_dict['test_acc_log_admm'] = [top1_test_accuracy(softmax(np.dot(X, log_dict['beta_log_admm'])), rankings_test)]
    log_dict['test_kendall_log_admm'] = [kendall_tau_test(softmax(np.dot(X, log_dict['beta_log_admm'])), rankings_test)]
    log_dict['iter_log_admm'] = 0
    # convex newton parameters
    log_dict['theta_newton_conv'] = False
    log_dict['mle_theta'] = ConvexPLModel(endog=endog, exog=exog, n=n)
    log_dict['theta_newton'] = np.copy(theta_init)
    log_dict['pi_newton'] = softmax(log_dict['theta_newton'])
    log_dict['time_newton_theta'] = [time_theta_init]
    log_dict['diff_pi_newton_theta'] = [np.linalg.norm(log_dict['pi_newton'])]
    log_dict['obj_newton_theta'] = [objective(log_dict['pi_newton'], rankings_train)]
    log_dict['test_acc_newton_theta'] = [top1_test_accuracy(log_dict['pi_newton'], rankings_test)]
    log_dict['test_kendall_newton_theta'] = [kendall_tau_test(log_dict['pi_newton'], rankings_test)]
    log_dict['iter_newton_theta'] = 0
    # unconstrained newton beta parameters
    log_dict['beta_newton_exp_beta_conv'] = False
    log_dict['mle_exp_beta'] = ExpBetaPLModel(endog=endog, exog=exog, X=X)
    log_dict['beta_newton_exp_beta'] = np.copy(exp_beta_init)
    log_dict['pi_newton_exp_beta'] = softmax(np.dot(X, log_dict['beta_newton_exp_beta']))
    log_dict['time_newton_exp_beta'] = [time_exp_beta_init]
    log_dict['diff_pi_newton_exp_beta'] = [np.linalg.norm(log_dict['pi_newton_exp_beta'])]
    log_dict['diff_beta_newton_exp_beta'] = [np.linalg.norm(log_dict['beta_newton_exp_beta'])]
    log_dict['obj_newton_exp_beta'] = [objective(log_dict['pi_newton_exp_beta'], rankings_train)]
    log_dict['test_acc_newton_exp_beta'] = [top1_test_accuracy(log_dict['pi_newton_exp_beta'], rankings_test)]
    log_dict['test_kendall_newton_exp_beta'] = [kendall_tau_test(log_dict['pi_newton_exp_beta'], rankings_test)]
    log_dict['iter_newton_exp_beta'] = 0
    # constrained slsqp beta parameters, normalize scores for comparison
    log_dict['slsqp_conv'] = False
    log_dict['mle_beta'] = LinBetaPLModel(endog=endog, exog=exog, X=X)
    log_dict['beta_slsqp'] = np.copy(beta_init)
    log_dict['b_slsqp'] = b_init
    log_dict['pi_slsqp'] = (np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp']) / np.sum(np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp'])
    log_dict['time_slsqp'] = [time_beta_b_init]
    log_dict['diff_pi_slsqp'] = [np.linalg.norm(log_dict['pi_slsqp'])]
    log_dict['diff_beta_slsqp'] = [np.linalg.norm(log_dict['beta_slsqp'])]
    log_dict['obj_slsqp'] = [objective(log_dict['pi_slsqp'], rankings_train)]
    log_dict['test_acc_slsqp'] = [top1_test_accuracy(log_dict['pi_slsqp'], rankings_test)]
    log_dict['test_kendall_slsqp'] = [kendall_tau_test(log_dict['pi_slsqp'], rankings_test)]
    log_dict['iter_slsqp'] = 0
    print('INITIALIZATION OVER!')
    return log_dict, save_name, rankings_train, X, rankings_test, mat_Pij, endog, exog


def run_save_all_methods_real_data(tasks, dir, d, test_fold, rho=1):
    '''
    Run all methods and save all logged results
    :param tasks: list of algorithms to be run
    :param dir: data directory
    :param d: number of full rankings
    :param test_fold: index of test fold
    :param rho: penalty parameter of ADMM
    '''
    log_dict, save_name, rankings_train, X, rankings_test, mat_Pij, endog, exog = init_all_methods_real_data(dir, d, test_fold)
    n = X.shape[0]  # number of items
    for iter in range(n_iter):
        # lsr update
        if not log_dict['lsr_conv'] and 'lsr' in tasks:
            log_dict['pi_lsr_prev'] = log_dict['pi_lsr']
            log_dict['pi_lsr'], time_lsr_iter = ilsr(n, rankings_train, weights=log_dict['pi_lsr'])
            log_dict['time_lsr'].append(time_lsr_iter)
            log_dict['diff_pi_lsr'].append(np.linalg.norm(log_dict['pi_lsr_prev'] - log_dict['pi_lsr']))
            log_dict['obj_lsr'].append(objective(log_dict['pi_lsr'], rankings_train))
            log_dict['test_acc_lsr'].append(top1_test_accuracy(log_dict['pi_lsr'], rankings_test))
            log_dict['test_kendall_lsr'].append(kendall_tau_test(log_dict['pi_lsr'], rankings_test))
            log_dict['iter_lsr'] += 1
            log_dict['lsr_conv'] = np.linalg.norm(log_dict['pi_lsr_prev'] - log_dict['pi_lsr']) < rtol * np.linalg.norm(log_dict['pi_lsr'])
        # mm update
        if not log_dict['mm_conv'] and 'mm' in tasks:
            log_dict['pi_mm_prev'] = log_dict['pi_mm']
            log_dict['pi_mm'], time_mm_iter = mm_iter(n, rankings_train, weights=log_dict['pi_mm'])
            if np.any(np.isnan(log_dict['pi_mm'])):
                log_dict['mm_conv'] = True
            else:
                log_dict['time_mm'].append(time_mm_iter)
                log_dict['diff_pi_mm'].append(np.linalg.norm(log_dict['pi_mm_prev'] - log_dict['pi_mm']))
                log_dict['obj_mm'].append(objective(log_dict['pi_mm'], rankings_train))
                log_dict['test_acc_mm'].append(top1_test_accuracy(log_dict['pi_mm'], rankings_test))
                log_dict['test_kendall_mm'].append(kendall_tau_test(log_dict['pi_mm'], rankings_test))
                log_dict['iter_mm'] += 1
                log_dict['mm_conv'] = np.linalg.norm(log_dict['pi_mm_prev'] - log_dict['pi_mm']) < rtol * np.linalg.norm(log_dict['pi_mm'])
        # lin_admm update
        if not log_dict['lin_admm_conv'] and 'lin_admm' in tasks:
            log_dict['pi_lin_admm_prev'] = log_dict['pi_lin_admm']
            log_dict['beta_lin_admm_prev'] = log_dict['beta_lin_admm']
            log_dict['tilde_pi_lin_admm_prev'] = (np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm']) / np.sum(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'])
            log_dict['pi_lin_admm'], log_dict['beta_lin_admm'], log_dict['b_lin_admm'], log_dict['u_lin_admm'], time_lin_admm_iter = \
                    log_dict['lin_admm'].fit_lin(rho, weights=log_dict['pi_lin_admm'], beta=log_dict['beta_lin_admm'], b=log_dict['b_lin_admm'], u=log_dict['u_lin_admm'])
            # scores predicted by beta
            log_dict['tilde_pi_lin_admm'] = (np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm']) / np.sum(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'])
            log_dict['time_lin_admm'].append(time_lin_admm_iter)
            log_dict['diff_pi_lin_admm'].append(np.linalg.norm(log_dict['pi_lin_admm_prev'] - log_dict['pi_lin_admm']))
            log_dict['diff_beta_lin_admm'].append(np.linalg.norm(log_dict['beta_lin_admm_prev'] - log_dict['beta_lin_admm']))
            log_dict['prim_feas_lin_admm'].append(np.linalg.norm(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'] - log_dict['pi_lin_admm']))
            log_dict['dual_feas_lin_admm'].append(np.linalg.norm(np.dot(log_dict['lin_admm'].X_tilde.T, log_dict['pi_lin_admm_prev'] - log_dict['pi_lin_admm'])))
            log_dict['obj_lin_admm'].append(objective(log_dict['pi_lin_admm'], rankings_train))
            log_dict['test_acc_lin_admm'].append(top1_test_accuracy(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'], rankings_test))
            log_dict['test_kendall_lin_admm'].append(kendall_tau_test(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm'], rankings_test))
            log_dict['iter_lin_admm'] += 1
            # log_dict['lin_admm_conv'] = log_dict['prim_feas_lin_admm'][-1] < rtol * np.max(
            #        [np.linalg.norm(np.dot(X, log_dict['beta_lin_admm']) + log_dict['b_lin_admm']),
            #        np.linalg.norm(log_dict['pi_lin_admm'])]) and log_dict['dual_feas_lin_admm'][-1] < \
            #        rtol * np.linalg.norm(np.dot(log_dict['lin_admm'].X_tilde.T, log_dict['u_lin_admm']))
            log_dict['lin_admm_conv'] = np.linalg.norm(
                    log_dict['pi_lin_admm_prev'] - log_dict['pi_lin_admm']) < rtol * np.linalg.norm(log_dict['pi_lin_admm']) \
                    and np.linalg.norm(log_dict['tilde_pi_lin_admm_prev'] - log_dict['tilde_pi_lin_admm']) < rtol * np.linalg.norm(
                    log_dict['tilde_pi_lin_admm'])
        # log_admm update
        if not log_dict['log_admm_conv'] and 'log_admm' in tasks:
            log_dict['pi_log_admm_prev'] = log_dict['pi_log_admm']
            log_dict['beta_log_admm_prev'] = log_dict['beta_log_admm']
            log_dict['tilde_pi_log_admm_prev'] = softmax(np.dot(X, log_dict['beta_log_admm']))
            log_dict['pi_log_admm'], log_dict['beta_log_admm'], log_dict['u_log_admm'], time_log_admm_iter = \
                    log_dict['log_admm'].fit_log(rho, weights=log_dict['pi_log_admm'], beta=log_dict['beta_log_admm'], u=log_dict['u_log_admm'])
            # scores predicted by beta
            log_dict['tilde_pi_log_admm'] = softmax(np.dot(X, log_dict['beta_log_admm']))
            log_dict['time_log_admm'].append(time_log_admm_iter)
            log_dict['diff_pi_log_admm'].append(np.linalg.norm(log_dict['pi_log_admm_prev'] - log_dict['pi_log_admm']))
            log_dict['diff_beta_log_admm'].append(np.linalg.norm(log_dict['beta_log_admm_prev'] - log_dict['beta_log_admm']))
            log_dict['prim_feas_log_admm'].append(np.linalg.norm(np.dot(X, log_dict['beta_log_admm']) - np.log(log_dict['pi_log_admm'] + epsilon)))
            log_dict['dual_feas_log_admm'].append(np.linalg.norm(np.dot(log_dict['log_admm'].X.T, np.log(log_dict['pi_log_admm_prev'] + epsilon) - np.log(log_dict['pi_log_admm'] + epsilon))))
            log_dict['obj_log_admm'].append(objective(log_dict['pi_log_admm'], rankings_train))
            log_dict['test_acc_log_admm'].append(top1_test_accuracy(softmax(np.dot(X, log_dict['beta_log_admm'])), rankings_test))
            log_dict['test_kendall_log_admm'].append(kendall_tau_test(softmax(np.dot(X, log_dict['beta_log_admm'])), rankings_test))
            log_dict['iter_log_admm'] += 1
            # log_dict['log_admm_conv'] = log_dict['prim_feas_log_admm'][-1] < rtol * np.max([np.linalg.norm(np.dot(X, log_dict['beta_log_admm'])),
            #        np.linalg.norm(np.log(log_dict['pi_log_admm'] + epsilon))]) and log_dict['dual_feas_log_admm'][-1] < \
            #        rtol * np.linalg.norm(np.dot(log_dict['log_admm'].X.T, log_dict['u_log_admm']))
            log_dict['log_admm_conv'] = np.linalg.norm(
                    log_dict['pi_log_admm_prev'] - log_dict['pi_log_admm']) < rtol * np.linalg.norm(log_dict['pi_log_admm']) \
                    and np.linalg.norm(log_dict['tilde_pi_log_admm_prev'] - log_dict['tilde_pi_log_admm']) < rtol * np.linalg.norm(
                    log_dict['tilde_pi_log_admm'])
        # convex newton update
        if not log_dict['theta_newton_conv'] and 'theta_newton' in tasks:
            try:
                log_dict['theta_newton_prev'] = log_dict['theta_newton']
                log_dict['pi_newton_prev'] = softmax(log_dict['theta_newton'])
                log_dict['theta_newton'], time_newton_theta_iter = log_dict['mle_theta'].fit(start_params=log_dict['theta_newton'][:-1], maxiter=1)
                log_dict['pi_newton'] = softmax(log_dict['theta_newton'])
                log_dict['time_newton_theta'].append(time_newton_theta_iter)
                log_dict['diff_pi_newton_theta'].append(np.linalg.norm(log_dict['pi_newton_prev'] - log_dict['pi_newton']))
                log_dict['obj_newton_theta'].append(objective(log_dict['pi_newton'], rankings_train))
                log_dict['test_acc_newton_theta'].append(top1_test_accuracy(log_dict['pi_newton'], rankings_test))
                log_dict['test_kendall_newton_theta'].append(kendall_tau_test(log_dict['pi_newton'], rankings_test))
                log_dict['iter_newton_theta'] += 1
                log_dict['theta_newton_conv'] = np.linalg.norm(log_dict['pi_newton_prev'] - log_dict['pi_newton']) < rtol * np.linalg.norm(log_dict['pi_newton'])
            except np.linalg.LinAlgError:
                print('Convex Newton diverged')
        # unconstrained newton beta parameters
        if not log_dict['beta_newton_exp_beta_conv'] and 'beta_newton_exp_beta' in tasks:
            try:
                log_dict['beta_newton_exp_beta_prev'] = log_dict['beta_newton_exp_beta']
                log_dict['pi_newton_exp_beta_prev'] = softmax(np.dot(X, log_dict['beta_newton_exp_beta']))
                log_dict['beta_newton_exp_beta'], time_newton_exp_beta_iter = log_dict['mle_exp_beta'].fit(params=log_dict['beta_newton_exp_beta'], max_iter=1)
                if np.any(np.isnan(log_dict['beta_newton_exp_beta'])):
                    log_dict['beta_newton_exp_beta_conv'] = True
                else:
                    log_dict['pi_newton_exp_beta'] = softmax(np.dot(X, log_dict['beta_newton_exp_beta']))
                    log_dict['time_newton_exp_beta'].append(time_newton_exp_beta_iter)
                    log_dict['diff_pi_newton_exp_beta'].append(np.linalg.norm(log_dict['pi_newton_exp_beta_prev'] - log_dict['pi_newton_exp_beta']))
                    log_dict['diff_beta_newton_exp_beta'].append(np.linalg.norm(log_dict['beta_newton_exp_beta_prev'] - log_dict['beta_newton_exp_beta']))
                    log_dict['obj_newton_exp_beta'].append(objective(log_dict['pi_newton_exp_beta'], rankings_train))
                    log_dict['test_acc_newton_exp_beta'].append(top1_test_accuracy(log_dict['pi_newton_exp_beta'], rankings_test))
                    log_dict['test_kendall_newton_exp_beta'].append(kendall_tau_test(log_dict['pi_newton_exp_beta'], rankings_test))
                    log_dict['iter_newton_exp_beta'] += 1
                    log_dict['beta_newton_exp_beta_conv'] = np.linalg.norm(log_dict['pi_newton_exp_beta_prev'] - log_dict['pi_newton_exp_beta']) < rtol * np.linalg.norm(log_dict['pi_newton_exp_beta'])
            except np.linalg.LinAlgError:
                print('Newton on Beta diverged')
        # slsqp update, normalize scores for comparison
        if not log_dict['slsqp_conv'] and 'slsqp' in tasks:
            log_dict['beta_slsqp_prev'] = log_dict['beta_slsqp']
            log_dict['b_slsqp_prev'] = log_dict['b_slsqp']
            log_dict['pi_slsqp_prev'] = (np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp']) / np.sum(np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp'])
            log_dict['beta_slsqp'], log_dict['b_slsqp'], time_slsqp_iter = log_dict['mle_beta'].fit(params=np.concatenate((log_dict['beta_slsqp'], [log_dict['b_slsqp']])),
                                                                method='SLSQP', max_iter=1)
            log_dict['pi_slsqp'] = (np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp']) / np.sum(np.dot(X, log_dict['beta_slsqp']) + log_dict['b_slsqp'])
            log_dict['time_slsqp'].append(time_slsqp_iter)
            log_dict['diff_pi_slsqp'].append(np.linalg.norm(log_dict['pi_slsqp_prev'] - log_dict['pi_slsqp']))
            log_dict['diff_beta_slsqp'].append(np.linalg.norm(log_dict['beta_slsqp_prev'] - log_dict['beta_slsqp']))
            log_dict['obj_slsqp'].append(objective(log_dict['pi_slsqp'], rankings_train))
            log_dict['test_acc_slsqp'].append(top1_test_accuracy(log_dict['pi_slsqp'], rankings_test))
            log_dict['test_kendall_slsqp'].append(kendall_tau_test(log_dict['pi_slsqp'], rankings_test))
            log_dict['iter_slsqp'] += 1
            log_dict['slsqp_conv'] = np.linalg.norm(log_dict['pi_slsqp_prev'] - log_dict['pi_slsqp']) < rtol * np.linalg.norm(log_dict['pi_slsqp'])
        # stop if all converged
        if log_dict['lsr_conv'] and log_dict['mm_conv'] and log_dict['lin_admm_conv'] and log_dict['log_admm_conv'] and log_dict['theta_newton_conv'] \
                    and log_dict['beta_newton_exp_beta_conv'] and log_dict['slsqp_conv']:
            break
    # Correct time scale
    log_dict['time_cont_lsr'] = [sum(log_dict['time_lsr'][:ind + 1]) for ind in range(len(log_dict['time_lsr']))]
    log_dict['time_cont_mm'] = [sum(log_dict['time_mm'][:ind + 1]) for ind in range(len(log_dict['time_mm']))]
    log_dict['time_cont_lin_admm'] = [sum(log_dict['time_lin_admm'][:ind + 1]) for ind in range(len(log_dict['time_lin_admm']))]
    log_dict['time_cont_log_admm'] = [sum(log_dict['time_log_admm'][:ind + 1]) for ind in range(len(log_dict['time_log_admm']))]
    log_dict['time_cont_newton_theta'] = [sum(log_dict['time_newton_theta'][:ind + 1]) for ind in range(len(log_dict['time_newton_theta']))]
    log_dict['time_cont_newton_exp_beta'] = [sum(log_dict['time_newton_exp_beta'][:ind + 1]) for ind in range(len(log_dict['time_newton_exp_beta']))]
    log_dict['time_cont_slsqp'] = [sum(log_dict['time_slsqp'][:ind + 1]) for ind in range(len(log_dict['time_slsqp']))]
    # Save results as a csv file
    save_name = save_name + '_rho_' + str(rho)
    with open('../results/' + dir + 'fig/' + '_logs_' + save_name + '.pickle', "wb") as pickle_out:
        pickle.dump(log_dict, pickle_out)
        pickle_out.close()
    return log_dict, save_name


def metric_and_CI(tasks, dir, save_name, n_fold=10):
    '''
    Load results for all folds and all methods. Write the averages and CIs to a csv file
    :param tasks: list of algorithms to be evaluated.
    :param dir: directory of log files
    :param save_name: fold_d_rho or fold_n_p_k_d_rho. CHANGE first char as the fold index
    '''
    results_dict = dict()
    # lin_admm
    results_dict['time_cont_lin_admm'] = []
    results_dict['iter_lin_admm'] = []
    results_dict['diff_pi_lin_admm'] = []
    results_dict['diff_beta_lin_admm'] = []
    results_dict['test_acc_lin_admm'] = []
    results_dict['test_kendall_lin_admm'] = []
    # log_admm
    results_dict['time_cont_log_admm'] = []
    results_dict['iter_log_admm'] = []
    results_dict['diff_pi_log_admm'] = []
    results_dict['diff_beta_log_admm'] = []
    results_dict['test_acc_log_admm'] = []
    results_dict['test_kendall_log_admm'] = []
    # lsr
    results_dict['time_cont_lsr'] = []
    results_dict['iter_lsr'] = []
    results_dict['diff_pi_lsr'] = []
    results_dict['test_acc_lsr'] = []
    results_dict['test_kendall_lsr'] = []
    # mm
    results_dict['time_cont_mm'] = []
    results_dict['iter_mm'] = []
    results_dict['diff_pi_mm'] = []
    results_dict['test_acc_mm'] = []
    results_dict['test_kendall_mm'] = []
    # convex newton
    results_dict['time_cont_newton_theta'] = []
    results_dict['iter_newton_theta'] = []
    results_dict['diff_pi_newton_theta'] = []
    results_dict['test_acc_newton_theta'] = []
    results_dict['test_kendall_newton_theta'] = []
    # newton on beta
    results_dict['time_cont_newton_exp_beta'] = []
    results_dict['iter_newton_exp_beta'] = []
    results_dict['diff_pi_newton_exp_beta'] = []
    results_dict['diff_beta_newton_exp_beta'] = []
    results_dict['test_acc_newton_exp_beta'] = []
    results_dict['test_kendall_newton_exp_beta'] = []
    # slsqp
    results_dict['time_cont_slsqp'] = []
    results_dict['iter_slsqp'] = []
    results_dict['diff_pi_slsqp'] = []
    results_dict['diff_beta_slsqp'] = []
    results_dict['test_acc_slsqp'] = []
    results_dict['test_kendall_slsqp'] = []
    # get the results at convergence for all folds and all methods
    for test_fold in range(n_fold):
        pos_fold = save_name.find('_')
        if exists('../results/' + dir + 'fig/' + '_logs_' + str(test_fold) + '_' + save_name[pos_fold + 1:] + '.pickle'):
            with open('../results/' + dir + 'fig/' + '_logs_' + str(test_fold) + '_' + save_name[pos_fold + 1:] + '.pickle', mode='rb') as pickle_in:
                log_dict = pickle.load(pickle_in)
            if 'lin_admm' in tasks:
                # lin_admm
                results_dict['time_cont_lin_admm'].append(log_dict['time_cont_lin_admm'][-1])
                results_dict['iter_lin_admm'].append(log_dict['iter_lin_admm'])
                results_dict['diff_pi_lin_admm'].append(log_dict['diff_pi_lin_admm'][-1])
                #results_dict['diff_beta_lin_admm'].append(log_dict['diff_beta_lin_admm'][-1])
                results_dict['test_acc_lin_admm'].append(log_dict['test_acc_lin_admm'][-1])
                results_dict['test_kendall_lin_admm'].append(log_dict['test_kendall_lin_admm'][-1])
            if 'log_admm' in tasks:
                # log_admm
                results_dict['time_cont_log_admm'].append(log_dict['time_cont_log_admm'][-1])
                results_dict['iter_log_admm'].append(log_dict['iter_log_admm'])
                results_dict['diff_pi_log_admm'].append(log_dict['diff_pi_log_admm'][-1])
                #results_dict['diff_beta_log_admm'].append(log_dict['diff_beta_log_admm'][-1])
                results_dict['test_acc_log_admm'].append(log_dict['test_acc_log_admm'][-1])
                results_dict['test_kendall_log_admm'].append(log_dict['test_kendall_log_admm'][-1])
            if 'lsr' in tasks:
                # lsr
                results_dict['time_cont_lsr'].append(log_dict['time_cont_lsr'][-1])
                results_dict['iter_lsr'].append(log_dict['iter_lsr'])
                results_dict['diff_pi_lsr'].append(log_dict['diff_pi_lsr'][-1])
                results_dict['test_acc_lsr'].append(log_dict['test_acc_lsr'][-1])
                results_dict['test_kendall_lsr'].append(log_dict['test_kendall_lsr'][-1])
            if 'mm' in tasks:
                # mm
                results_dict['time_cont_mm'].append(log_dict['time_cont_mm'][-1])
                results_dict['iter_mm'].append(log_dict['iter_mm'])
                results_dict['diff_pi_mm'].append(log_dict['diff_pi_mm'][-1])
                results_dict['test_acc_mm'].append(log_dict['test_acc_mm'][-1])
                results_dict['test_kendall_mm'].append(log_dict['test_kendall_mm'][-1])
            if 'theta_newton' in tasks:
                # convex newton
                results_dict['time_cont_newton_theta'].append(log_dict['time_cont_newton_theta'][-1])
                results_dict['iter_newton_theta'].append(log_dict['iter_newton_theta'])
                results_dict['diff_pi_newton_theta'].append(log_dict['diff_pi_newton_theta'][-1])
                results_dict['test_acc_newton_theta'].append(log_dict['test_acc_newton_theta'][-1])
                results_dict['test_kendall_newton_theta'].append(log_dict['test_kendall_newton_theta'][-1])
            if 'beta_newton_exp_beta' in tasks:
                # newton on beta
                results_dict['time_cont_newton_exp_beta'].append(log_dict['time_cont_newton_exp_beta'][-1])
                results_dict['iter_newton_exp_beta'].append(log_dict['iter_newton_exp_beta'])
                results_dict['diff_pi_newton_exp_beta'].append(log_dict['diff_pi_newton_exp_beta'][-1])
                #results_dict['diff_beta_newton_exp_beta'].append(log_dict['diff_beta_newton_exp_beta'][-1])
                results_dict['test_acc_newton_exp_beta'].append(log_dict['test_acc_newton_exp_beta'][-1])
                results_dict['test_kendall_newton_exp_beta'].append(log_dict['test_kendall_newton_exp_beta'][-1])
            if 'slsqp' in tasks:
                # slsqp
                results_dict['time_cont_slsqp'].append(log_dict['time_cont_slsqp'][-1])
                results_dict['iter_slsqp'].append(log_dict['iter_slsqp'])
                results_dict['diff_pi_slsqp'].append(log_dict['diff_pi_slsqp'][-1])
                #results_dict['diff_beta_slsqp'].append(log_dict['diff_beta_slsqp'][-1])
                results_dict['test_acc_slsqp'].append(log_dict['test_acc_slsqp'][-1])
                results_dict['test_kendall_slsqp'].append(log_dict['test_kendall_slsqp'][-1])
    # find average and CI results for all methods
    if 'lin_admm' in tasks:
        # lin_admm
        results_dict['time_cont_lin_admm'] = '$' + str(ceil(np.mean(results_dict['time_cont_lin_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_lin_admm']) * 1000) / 1000) + '$'
        results_dict['iter_lin_admm'] = '$' + str(ceil(np.mean(results_dict['iter_lin_admm']))) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['iter_lin_admm']))) + '$'
        results_dict['diff_pi_lin_admm'] = '$' + str(ceil(np.mean(results_dict['diff_pi_lin_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_lin_admm']) * 1000) / 1000) + '$'
        #results_dict['diff_beta_lin_admm'] = '$' + str(ceil(np.mean(results_dict['diff_beta_lin_admm']) * 1000) / 1000) + ' \pm ' + str(ceil(np.std(results_dict['diff_beta_lin_admm']) * 1000) / 1000) + '$'
        results_dict['test_acc_lin_admm'] = '$' + str(ceil(np.mean(results_dict['test_acc_lin_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_acc_lin_admm']) * 1000) / 1000) + '$'
        results_dict['test_kendall_lin_admm'] = '$' + str(ceil(np.mean(results_dict['test_kendall_lin_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_lin_admm']) * 1000) / 1000) + '$'
    if 'log_admm' in tasks:
        # log_admm
        results_dict['time_cont_log_admm'] = '$' + str(ceil(np.mean(results_dict['time_cont_log_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_log_admm']) * 1000) / 1000) + '$'
        results_dict['iter_log_admm'] = '$' + str(ceil(np.mean(results_dict['iter_log_admm']))) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['iter_log_admm']))) + '$'
        results_dict['diff_pi_log_admm'] = '$' + str(ceil(np.mean(results_dict['diff_pi_log_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_log_admm']) * 1000) / 1000) + '$'
        #results_dict['diff_beta_log_admm'] = '$' + str(ceil(np.mean(results_dict['diff_beta_log_admm']) * 1000) / 1000) + ' \pm ' + str(ceil(np.std(results_dict['diff_beta_log_admm']) * 1000) / 1000) + '$'
        results_dict['test_acc_log_admm'] = '$' + str(ceil(np.mean(results_dict['test_acc_log_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_acc_log_admm']) * 1000) / 1000) + '$'
        results_dict['test_kendall_log_admm'] = '$' + str(ceil(np.mean(results_dict['test_kendall_log_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_log_admm']) * 1000) / 1000) + '$'
    if 'lsr' in tasks:
        # lsr
        results_dict['time_cont_lsr'] = '$' + str(ceil(np.mean(results_dict['time_cont_lsr']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_lsr']) * 1000) / 1000) + '$'
        results_dict['iter_lsr'] = '$' + str(ceil(np.mean(results_dict['iter_lsr']))) + ' \pm ' + \
                                   str(ceil(np.std(results_dict['iter_lsr']))) + '$'
        results_dict['diff_pi_lsr'] = '$' + str(ceil(np.mean(results_dict['diff_pi_lsr']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_lsr']) * 1000) / 1000) + '$'
        results_dict['test_acc_lsr'] = '$' + str(ceil(np.mean(results_dict['test_acc_lsr']) * 1000) / 1000) + ' \pm ' + \
                                       str(ceil(np.std(results_dict['test_acc_lsr']) * 1000) / 1000) + '$'
        results_dict['test_kendall_lsr'] = '$' + str(ceil(np.mean(results_dict['test_kendall_lsr']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_lsr']) * 1000) / 1000) + '$'
    if 'mm' in tasks:
        # mm
        results_dict['time_cont_mm'] = '$' + str(ceil(np.mean(results_dict['time_cont_mm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_mm']) * 1000) / 1000) + '$'
        results_dict['iter_mm'] = '$' + str(ceil(np.mean(results_dict['iter_mm']))) + ' \pm ' + \
                                   str(ceil(np.std(results_dict['iter_mm']))) + '$'
        results_dict['diff_pi_mm'] = '$' + str(ceil(np.mean(results_dict['diff_pi_mm']) * 1000) / 1000) + ' \pm ' + \
                                      str(ceil(np.std(results_dict['diff_pi_mm']) * 1000) / 1000) + '$'
        results_dict['test_acc_mm'] = '$' + str(ceil(np.mean(results_dict['test_acc_mm']) * 1000) / 1000) + ' \pm ' + \
                                       str(ceil(np.std(results_dict['test_acc_mm']) * 1000) / 1000) + '$'
        results_dict['test_kendall_mm'] = '$' + str(ceil(np.mean(results_dict['test_kendall_mm']) * 1000) / 1000) + ' \pm ' + \
                                           str(ceil(np.std(results_dict['test_kendall_mm']) * 1000) / 1000) + '$'
    if 'theta_newton' in tasks:
        # convex newton
        results_dict['time_cont_newton_theta'] = '$' + str(ceil(np.mean(results_dict['time_cont_newton_theta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_newton_theta']) * 1000) / 1000) + '$'
        results_dict['iter_newton_theta'] = '$' + str(ceil(np.mean(results_dict['iter_newton_theta']))) + ' \pm ' + \
                                            str(ceil(np.std(results_dict['iter_newton_theta']))) + '$'
        results_dict['diff_pi_newton_theta'] = '$' + str(ceil(np.mean(results_dict['diff_pi_newton_theta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_newton_theta']) * 1000) / 1000) + '$'
        results_dict['test_acc_newton_theta'] = '$' + str(ceil(np.mean(results_dict['test_acc_newton_theta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_acc_newton_theta']) * 1000) / 1000) + '$'
        results_dict['test_kendall_newton_theta'] = '$' + str(ceil(np.mean(results_dict['test_kendall_newton_theta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_newton_theta']) * 1000) / 1000) + '$'
    if 'beta_newton_exp_beta' in tasks:
        # newton on beta
        results_dict['time_cont_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['time_cont_newton_exp_beta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_newton_exp_beta']) * 1000) / 1000) + '$'
        results_dict['iter_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['iter_newton_exp_beta']))) + ' \pm ' + \
                                               str(ceil(np.std(results_dict['iter_newton_exp_beta']))) + '$'
        results_dict['diff_pi_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['diff_pi_newton_exp_beta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_newton_exp_beta']) * 1000) / 1000) + '$'
        #results_dict['diff_beta_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['diff_beta_newton_exp_beta']) * 1000) / 1000) + ' \pm ' + str(ceil(np.std(results_dict['diff_beta_newton_exp_beta']) * 1000) / 1000) + '$'
        results_dict['test_acc_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['test_acc_newton_exp_beta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_acc_newton_exp_beta']) * 1000) / 1000) + '$'
        results_dict['test_kendall_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['test_kendall_newton_exp_beta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_newton_exp_beta']) * 1000) / 1000) + '$'
    if 'slsqp' in tasks:
        # slsqp
        results_dict['time_cont_slsqp'] = '$' + str(ceil(np.mean(results_dict['time_cont_slsqp']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_slsqp']) * 1000) / 1000) + '$'
        results_dict['iter_slsqp'] = '$' + str(ceil(np.mean(results_dict['iter_slsqp']))) + ' \pm ' + \
                                     str(ceil(np.std(results_dict['iter_slsqp']))) + '$'
        results_dict['diff_pi_slsqp'] = '$' + str(ceil(np.mean(results_dict['diff_pi_slsqp']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_slsqp']) * 1000) / 1000) + '$'
        #results_dict['diff_beta_slsqp'] = '$' + str(ceil(np.mean(results_dict['diff_beta_slsqp']) * 1000) / 1000) + ' \pm ' + str(ceil(np.std(results_dict['diff_beta_slsqp']) * 1000) / 1000) + '$'
        results_dict['test_acc_slsqp'] = '$' + str(ceil(np.mean(results_dict['test_acc_slsqp']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_acc_slsqp']) * 1000) / 1000) + '$'
        results_dict['test_kendall_slsqp'] = '$' + str(ceil(np.mean(results_dict['test_kendall_slsqp']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_slsqp']) * 1000) / 1000) + '$'
    # Save results as a csv file
    with open('../results/' + dir + 'fig/' + '_results_' + save_name[2:] + '.csv', "w") as infile:
        w = csv.writer(infile)
        for key, val in results_dict.items():
            w.writerow([key, val])
    return results_dict


def metric_and_CI_over_seeds(tasks, dir, save_name, n_seeds, n_fold=10):
    '''
    Load results for all folds and all methods. Write the averages and CIs to a csv file.
    :param tasks: list of algorithms to be evaluated.
    :param dir: directory of log files
    :param save_name: fold_d_rho or fold_n_p_k_d_rho. CHANGE first char as the fold index
    '''
    results_dict = dict()
    # lin_admm
    results_dict['time_cont_lin_admm'] = []
    results_dict['iter_lin_admm'] = []
    results_dict['diff_pi_lin_admm'] = []
    results_dict['diff_beta_lin_admm'] = []
    results_dict['test_acc_lin_admm'] = []
    results_dict['test_kendall_lin_admm'] = []
    # log_admm
    results_dict['time_cont_log_admm'] = []
    results_dict['iter_log_admm'] = []
    results_dict['diff_pi_log_admm'] = []
    results_dict['diff_beta_log_admm'] = []
    results_dict['test_acc_log_admm'] = []
    results_dict['test_kendall_log_admm'] = []
    # lsr
    results_dict['time_cont_lsr'] = []
    results_dict['iter_lsr'] = []
    results_dict['diff_pi_lsr'] = []
    results_dict['test_acc_lsr'] = []
    results_dict['test_kendall_lsr'] = []
    # mm
    results_dict['time_cont_mm'] = []
    results_dict['iter_mm'] = []
    results_dict['diff_pi_mm'] = []
    results_dict['test_acc_mm'] = []
    results_dict['test_kendall_mm'] = []
    # convex newton
    results_dict['time_cont_newton_theta'] = []
    results_dict['iter_newton_theta'] = []
    results_dict['diff_pi_newton_theta'] = []
    results_dict['test_acc_newton_theta'] = []
    results_dict['test_kendall_newton_theta'] = []
    # newton on beta
    results_dict['time_cont_newton_exp_beta'] = []
    results_dict['iter_newton_exp_beta'] = []
    results_dict['diff_pi_newton_exp_beta'] = []
    results_dict['diff_beta_newton_exp_beta'] = []
    results_dict['test_acc_newton_exp_beta'] = []
    results_dict['test_kendall_newton_exp_beta'] = []
    # slsqp
    results_dict['time_cont_slsqp'] = []
    results_dict['iter_slsqp'] = []
    results_dict['diff_pi_slsqp'] = []
    results_dict['diff_beta_slsqp'] = []
    results_dict['test_acc_slsqp'] = []
    results_dict['test_kendall_slsqp'] = []
    # get the results at convergence for all folds and all methods
    for rand_iter in range(n_seeds):
        for test_fold in range(n_fold):
            pos_rand_iter = save_name.find('_')
            if exists('../results/' + dir + 'fig/' + '_logs_' + str(rand_iter) + '_' + str(test_fold) + save_name[pos_rand_iter+2:] + '.pickle'):
                with open('../results/' + dir + 'fig/' + '_logs_' + str(rand_iter) + '_' + str(test_fold) + save_name[pos_rand_iter+2:] + '.pickle', mode='rb') as pickle_in:
                    log_dict = pickle.load(pickle_in)
                if 'lin_admm' in tasks:
                    # lin_admm
                    results_dict['time_cont_lin_admm'].append(log_dict['time_cont_lin_admm'][-1])
                    results_dict['iter_lin_admm'].append(log_dict['iter_lin_admm'])
                    results_dict['diff_pi_lin_admm'].append(log_dict['diff_pi_lin_admm'][-1])
                    #results_dict['diff_beta_lin_admm'].append(log_dict['diff_beta_lin_admm'][-1])
                    results_dict['test_acc_lin_admm'].append(log_dict['test_acc_lin_admm'][-1])
                    results_dict['test_kendall_lin_admm'].append(log_dict['test_kendall_lin_admm'][-1])
                if 'log_admm' in tasks:
                    # log_admm
                    results_dict['time_cont_log_admm'].append(log_dict['time_cont_log_admm'][-1])
                    results_dict['iter_log_admm'].append(log_dict['iter_log_admm'])
                    results_dict['diff_pi_log_admm'].append(log_dict['diff_pi_log_admm'][-1])
                    #results_dict['diff_beta_log_admm'].append(log_dict['diff_beta_log_admm'][-1])
                    results_dict['test_acc_log_admm'].append(log_dict['test_acc_log_admm'][-1])
                    results_dict['test_kendall_log_admm'].append(log_dict['test_kendall_log_admm'][-1])
                if 'lsr' in tasks:
                    # lsr
                    results_dict['time_cont_lsr'].append(log_dict['time_cont_lsr'][-1])
                    results_dict['iter_lsr'].append(log_dict['iter_lsr'])
                    results_dict['diff_pi_lsr'].append(log_dict['diff_pi_lsr'][-1])
                    results_dict['test_acc_lsr'].append(log_dict['test_acc_lsr'][-1])
                    results_dict['test_kendall_lsr'].append(log_dict['test_kendall_lsr'][-1])
                if 'mm' in tasks:
                    # mm
                    results_dict['time_cont_mm'].append(log_dict['time_cont_mm'][-1])
                    results_dict['iter_mm'].append(log_dict['iter_mm'])
                    results_dict['diff_pi_mm'].append(log_dict['diff_pi_mm'][-1])
                    results_dict['test_acc_mm'].append(log_dict['test_acc_mm'][-1])
                    results_dict['test_kendall_mm'].append(log_dict['test_kendall_mm'][-1])
                if 'theta_newton' in tasks:
                    # convex newton
                    results_dict['time_cont_newton_theta'].append(log_dict['time_cont_newton_theta'][-1])
                    results_dict['iter_newton_theta'].append(log_dict['iter_newton_theta'])
                    results_dict['diff_pi_newton_theta'].append(log_dict['diff_pi_newton_theta'][-1])
                    results_dict['test_acc_newton_theta'].append(log_dict['test_acc_newton_theta'][-1])
                    results_dict['test_kendall_newton_theta'].append(log_dict['test_kendall_newton_theta'][-1])
                if 'beta_newton_exp_beta' in tasks:
                    # newton on beta
                    results_dict['time_cont_newton_exp_beta'].append(log_dict['time_cont_newton_exp_beta'][-1])
                    results_dict['iter_newton_exp_beta'].append(log_dict['iter_newton_exp_beta'])
                    results_dict['diff_pi_newton_exp_beta'].append(log_dict['diff_pi_newton_exp_beta'][-1])
                    #results_dict['diff_beta_newton_exp_beta'].append(log_dict['diff_beta_newton_exp_beta'][-1])
                    results_dict['test_acc_newton_exp_beta'].append(log_dict['test_acc_newton_exp_beta'][-1])
                    results_dict['test_kendall_newton_exp_beta'].append(log_dict['test_kendall_newton_exp_beta'][-1])
                if 'slsqp' in tasks:
                    # slsqp
                    results_dict['time_cont_slsqp'].append(log_dict['time_cont_slsqp'][-1])
                    results_dict['iter_slsqp'].append(log_dict['iter_slsqp'])
                    results_dict['diff_pi_slsqp'].append(log_dict['diff_pi_slsqp'][-1])
                    #results_dict['diff_beta_slsqp'].append(log_dict['diff_beta_slsqp'][-1])
                    results_dict['test_acc_slsqp'].append(log_dict['test_acc_slsqp'][-1])
                    results_dict['test_kendall_slsqp'].append(log_dict['test_kendall_slsqp'][-1])
    # find average and CI results for all methods
    if 'lin_admm' in tasks:
        # lin_admm
        results_dict['time_cont_lin_admm'] = '$' + str(ceil(np.mean(results_dict['time_cont_lin_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_lin_admm']) * 1000) / 1000) + '$'
        results_dict['iter_lin_admm'] = '$' + str(ceil(np.mean(results_dict['iter_lin_admm']))) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['iter_lin_admm']))) + '$'
        results_dict['diff_pi_lin_admm'] = '$' + str(ceil(np.mean(results_dict['diff_pi_lin_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_lin_admm']) * 1000) / 1000) + '$'
        #results_dict['diff_beta_lin_admm'] = '$' + str(ceil(np.mean(results_dict['diff_beta_lin_admm']) * 1000) / 1000) + ' \pm ' + str(ceil(np.std(results_dict['diff_beta_lin_admm']) * 1000) / 1000) + '$'
        results_dict['test_acc_lin_admm'] = '$' + str(ceil(np.mean(results_dict['test_acc_lin_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_acc_lin_admm']) * 1000) / 1000) + '$'
        results_dict['test_kendall_lin_admm'] = '$' + str(ceil(np.mean(results_dict['test_kendall_lin_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_lin_admm']) * 1000) / 1000) + '$'
    if 'log_admm' in tasks:
        # log_admm
        results_dict['time_cont_log_admm'] = '$' + str(ceil(np.mean(results_dict['time_cont_log_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_log_admm']) * 1000) / 1000) + '$'
        results_dict['iter_log_admm'] = '$' + str(ceil(np.mean(results_dict['iter_log_admm']))) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['iter_log_admm']))) + '$'
        results_dict['diff_pi_log_admm'] = '$' + str(ceil(np.mean(results_dict['diff_pi_log_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_log_admm']) * 1000) / 1000) + '$'
        #results_dict['diff_beta_log_admm'] = '$' + str(ceil(np.mean(results_dict['diff_beta_log_admm']) * 1000) / 1000) + ' \pm ' + str(ceil(np.std(results_dict['diff_beta_log_admm']) * 1000) / 1000) + '$'
        results_dict['test_acc_log_admm'] = '$' + str(ceil(np.mean(results_dict['test_acc_log_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_acc_log_admm']) * 1000) / 1000) + '$'
        results_dict['test_kendall_log_admm'] = '$' + str(ceil(np.mean(results_dict['test_kendall_log_admm']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_log_admm']) * 1000) / 1000) + '$'
    if 'lsr' in tasks:
        # lsr
        results_dict['time_cont_lsr'] = '$' + str(ceil(np.mean(results_dict['time_cont_lsr']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_lsr']) * 1000) / 1000) + '$'
        results_dict['iter_lsr'] = '$' + str(ceil(np.mean(results_dict['iter_lsr']))) + ' \pm ' + \
                                   str(ceil(np.std(results_dict['iter_lsr']))) + '$'
        results_dict['diff_pi_lsr'] = '$' + str(ceil(np.mean(results_dict['diff_pi_lsr']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_lsr']) * 1000) / 1000) + '$'
        results_dict['test_acc_lsr'] = '$' + str(ceil(np.mean(results_dict['test_acc_lsr']) * 1000) / 1000) + ' \pm ' + \
                                       str(ceil(np.std(results_dict['test_acc_lsr']) * 1000) / 1000) + '$'
        results_dict['test_kendall_lsr'] = '$' + str(ceil(np.mean(results_dict['test_kendall_lsr']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_lsr']) * 1000) / 1000) + '$'
    if 'mm' in tasks:
        # mm
        results_dict['time_cont_mm'] = '$' + str(ceil(np.mean(results_dict['time_cont_mm']) * 1000) / 1000) + ' \pm ' + \
                                       str(ceil(np.std(results_dict['time_cont_mm']) * 1000) / 1000) + '$'
        results_dict['iter_mm'] = '$' + str(ceil(np.mean(results_dict['iter_mm']))) + ' \pm ' + \
                                  str(ceil(np.std(results_dict['iter_mm']))) + '$'
        results_dict['diff_pi_mm'] = '$' + str(ceil(np.mean(results_dict['diff_pi_mm']) * 1000) / 1000) + ' \pm ' + \
                                         str(ceil(np.std(results_dict['diff_pi_mm']) * 1000) / 1000) + '$'
        results_dict['test_acc_mm'] = '$' + str(ceil(np.mean(results_dict['test_acc_mm']) * 1000) / 1000) + ' \pm ' + \
                                      str(ceil(np.std(results_dict['test_acc_mm']) * 1000) / 1000) + '$'
        results_dict['test_kendall_mm'] = '$' + str(ceil(np.mean(results_dict['test_kendall_mm']) * 1000) / 1000) + ' \pm ' + \
                                          str(ceil(np.std(results_dict['test_kendall_mm']) * 1000) / 1000) + '$'
    if 'theta_newton' in tasks:
        # convex newton
        results_dict['time_cont_newton_theta'] = '$' + str(ceil(np.mean(results_dict['time_cont_newton_theta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_newton_theta']) * 1000) / 1000) + '$'
        results_dict['iter_newton_theta'] = '$' + str(ceil(np.mean(results_dict['iter_newton_theta']))) + ' \pm ' + \
                                            str(ceil(np.std(results_dict['iter_newton_theta']))) + '$'
        results_dict['diff_pi_newton_theta'] = '$' + str(ceil(np.mean(results_dict['diff_pi_newton_theta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_newton_theta']) * 1000) / 1000) + '$'
        results_dict['test_acc_newton_theta'] = '$' + str(ceil(np.mean(results_dict['test_acc_newton_theta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_acc_newton_theta']) * 1000) / 1000) + '$'
        results_dict['test_kendall_newton_theta'] = '$' + str(ceil(np.mean(results_dict['test_kendall_newton_theta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_newton_theta']) * 1000) / 1000) + '$'
    if 'beta_newton_exp_beta' in tasks:
        # newton on beta
        results_dict['time_cont_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['time_cont_newton_exp_beta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_newton_exp_beta']) * 1000) / 1000) + '$'
        results_dict['iter_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['iter_newton_exp_beta']))) + ' \pm ' + \
                                               str(ceil(np.std(results_dict['iter_newton_exp_beta']))) + '$'
        results_dict['diff_pi_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['diff_pi_newton_exp_beta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_newton_exp_beta']) * 1000) / 1000) + '$'
        #results_dict['diff_beta_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['diff_beta_newton_exp_beta']) * 1000) / 1000) + ' \pm ' + str(ceil(np.std(results_dict['diff_beta_newton_exp_beta']) * 1000) / 1000) + '$'
        results_dict['test_acc_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['test_acc_newton_exp_beta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_acc_newton_exp_beta']) * 1000) / 1000) + '$'
        results_dict['test_kendall_newton_exp_beta'] = '$' + str(ceil(np.mean(results_dict['test_kendall_newton_exp_beta']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_newton_exp_beta']) * 1000) / 1000) + '$'
    if 'slsqp' in tasks:
        # slsqp
        results_dict['time_cont_slsqp'] = '$' + str(ceil(np.mean(results_dict['time_cont_slsqp']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['time_cont_slsqp']) * 1000) / 1000) + '$'
        results_dict['iter_slsqp'] = '$' + str(ceil(np.mean(results_dict['iter_slsqp']))) + ' \pm ' + \
                                     str(ceil(np.std(results_dict['iter_slsqp']))) + '$'
        results_dict['diff_pi_slsqp'] = '$' + str(ceil(np.mean(results_dict['diff_pi_slsqp']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['diff_pi_slsqp']) * 1000) / 1000) + '$'
        #results_dict['diff_beta_slsqp'] = '$' + str(ceil(np.mean(results_dict['diff_beta_slsqp']) * 1000) / 1000) + ' \pm ' + str(ceil(np.std(results_dict['diff_beta_slsqp']) * 1000) / 1000) + '$'
        results_dict['test_acc_slsqp'] = '$' + str(ceil(np.mean(results_dict['test_acc_slsqp']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_acc_slsqp']) * 1000) / 1000) + '$'
        results_dict['test_kendall_slsqp'] = '$' + str(ceil(np.mean(results_dict['test_kendall_slsqp']) * 1000) / 1000) + ' \pm ' + \
                                        str(ceil(np.std(results_dict['test_kendall_slsqp']) * 1000) / 1000) + '$'
    # Save results as a csv file
    with open('../results/' + dir + 'fig/' + '_results_' + save_name[2:] + '.csv', "w") as infile:
        w = csv.writer(infile)
        for key, val in results_dict.items():
            w.writerow([key, val])
    return results_dict


if __name__ == "__main__":
    n_fold = 10
    n_seeds = 5
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # list of algorithms to be run OR 'test' if saving metrics and CI
    parser.add_argument('tasks', help='delimited list of tasks', type=lambda s: [str(task) for task in s.split(',')])
    parser.add_argument('test_fold', type=int)
    parser.add_argument('d', type=int)
    parser.add_argument('dir', type=str)
    # Dataset
    synthetic = True
    real_data = False
    # Algorithm parameters
    rho = 1  # tune for each data
    if synthetic:
        parser.add_argument('n', type=int)
        parser.add_argument('p', type=int)
        parser.add_argument('k', type=int)
        parser.add_argument('rand_iter', type=int)
        args = parser.parse_args()
        test_fold = args.test_fold
        rand_iter = args.rand_iter
        n = args.n
        p = args.p
        k = args.k
        d = args.d
        dir = args.dir
        if 'test' not in args.tasks:
            log_dict, save_name = run_save_all_methods_synthetic(args.tasks, dir, n, p, k, d, rand_iter, test_fold, rho=rho)
        else:
            save_name = str(rand_iter) + '_' + str(test_fold) + '_n_' + str(n) + '_p_' + str(p) + '_k_' + str(k) + '_d_' + str(d) + '_rho_' + str(rho)
            metric_and_CI_over_seeds(args.tasks, dir, save_name, n_seeds=n_seeds, n_fold=n_fold)
    elif real_data:
        args = parser.parse_args()
        test_fold = args.test_fold
        d = args.d
        dir = args.dir
        if 'test' not in args.tasks:
            log_dict, save_name = run_save_all_methods_real_data(args.tasks, dir, d, test_fold, rho=rho)
        else:
            save_name = str(test_fold) + '_d_' + str(d) + '_rho_' + str(rho)
            metric_and_CI(args.tasks, dir, save_name, n_fold=n_fold)


