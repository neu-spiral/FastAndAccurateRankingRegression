import matplotlib.pyplot as plt
from pylab import *
from run_methods import *

def save_fig():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xscale('log', basex=10)
    linestyle = {"linestyle": 'solid', "linewidth": 2, "markeredgewidth": 2}
    plt.plot(xaxis, mean_lin_admm, color='r', **linestyle)
    plt.fill_between(xaxis, mean_lin_admm - CI_lin_admm, mean_lin_admm + CI_lin_admm, color='r', alpha=0.05)
    linestyle = {"linestyle": 'dashed', "linewidth": 2, "markeredgewidth": 2}
    plt.plot(xaxis, mean_log_admm, color='k', **linestyle)
    plt.fill_between(xaxis, np.clip(mean_log_admm - CI_log_admm, a_min=mean_log_admm/10, a_max=None), mean_log_admm + CI_log_admm, color='k', alpha=0.05)
    linestyle = {"linestyle": 'dashdot', "linewidth": 2, "markeredgewidth": 2}
    plt.plot(xaxis, mean_ilsr, color='g', **linestyle)
    plt.fill_between(xaxis, mean_ilsr - CI_ilsr, mean_ilsr + CI_ilsr, color='g', alpha=0.05)
    linestyle = {"linestyle": 'dotted', "linewidth": 2, "markeredgewidth": 2}
    plt.plot(xaxis, mean_newton_on_beta, color='b', **linestyle)
    plt.fill_between(xaxis, mean_newton_on_beta - CI_newton_on_beta, mean_newton_on_beta + CI_newton_on_beta, color='b', alpha=0.05)
    ax.tick_params(labelsize='large')
    ax.legend(['PLADMM', 'PLADMM-log', 'ILSR', 'Newton on $\\beta$'], prop={'size': 18})
    #ax.set_yscale('symlog', basey=10)
    #ax.set_ylim(bottom=0)
    if 'Time' in y_label:
        ax.set_yscale('log', basey=10)
        plt.savefig('../results/' + dir + 'fig/syn_conv_time' + '_vs_' + x_label + '.pdf', bbox_inches='tight')
    elif 'Acc.' in y_label:
        plt.savefig('../results/' + dir + 'fig/syn_test_acc' + '_vs_' + x_label + '.pdf', bbox_inches='tight')

dir = 'synthetic_'
rand_iter = 0
test_fold = 0
n_fold = 10
rho = 1
###########################
n_range = [50,100,1000,10000,100000,1000000]
p_ntest = 100
M_ntest = 250
xaxis = n_range
x_label = 'n'
######## Conv Time vs n
mean_lin_admm = []
CI_lin_admm = []
mean_log_admm = []
CI_log_admm = []
mean_ilsr = []
CI_ilsr = []
mean_newton_on_beta = []
CI_newton_on_beta = []
for n in n_range:
    save_name = str(rand_iter) + '_' + str(test_fold) + '_n_' + str(n) + '_p_' + str(p_ntest) + '_k_' + str(2) + '_d_' + str(M_ntest) + '_rho_' + str(rho)
    results_dict = metric_and_CI_over_seeds(['test', 'lsr', 'mm', 'lin_admm', 'log_admm', 'theta_newton',
                                             'beta_newton_exp_beta', 'slsqp'], dir, save_name, n_seeds=n_fold,
                                            n_fold=n_fold)  # key:mean, val:CI
    mean_lin_admm.append(float(results_dict['time_cont_lin_admm'].replace("$", "").split(' ')[0]))
    CI_lin_admm.append(float(results_dict['time_cont_lin_admm'].replace("$", "").split(' ')[2]))
    mean_log_admm.append(float(results_dict['time_cont_log_admm'].replace("$", "").split(' ')[0]))
    CI_log_admm.append(float(results_dict['time_cont_log_admm'].replace("$", "").split(' ')[2]))
    mean_ilsr.append(float(results_dict['time_cont_lsr'].replace("$", "").split(' ')[0]))
    CI_ilsr.append(float(results_dict['time_cont_lsr'].replace("$", "").split(' ')[2]))
    mean_newton_on_beta.append(float(results_dict['time_cont_newton_exp_beta'].replace("$", "").split(' ')[0]))
    CI_newton_on_beta.append(float(results_dict['time_cont_newton_exp_beta'].replace("$", "").split(' ')[2]))
mean_lin_admm = np.array(mean_lin_admm)
CI_lin_admm = np.array(CI_lin_admm)
mean_log_admm = np.array(mean_log_admm)
CI_log_admm = np.array(CI_log_admm)
mean_ilsr = np.array(mean_ilsr)
CI_ilsr = np.array(CI_ilsr)
mean_newton_on_beta = np.array(mean_newton_on_beta)
CI_newton_on_beta = np.array(CI_newton_on_beta)
y_label = 'Time'
save_fig()
######## Accuracy vs n
mean_lin_admm = []
CI_lin_admm = []
mean_log_admm = []
CI_log_admm = []
mean_ilsr = []
CI_ilsr = []
mean_newton_on_beta = []
CI_newton_on_beta = []
for n in n_range:
    save_name = str(rand_iter) + '_' + str(test_fold) + '_n_' + str(n) + '_p_' + str(p_ntest) + '_k_' + str(2) + '_d_' + str(M_ntest) + '_rho_' + str(rho)
    results_dict = metric_and_CI_over_seeds(['test', 'lsr', 'mm', 'lin_admm', 'log_admm', 'theta_newton',
                                             'beta_newton_exp_beta', 'slsqp'], dir, save_name, n_seeds=n_fold,
                                            n_fold=n_fold)  # key:mean, val:CI
    mean_lin_admm.append(float(results_dict['test_acc_lin_admm'].replace("$", "").split(' ')[0]))
    CI_lin_admm.append(float(results_dict['test_acc_lin_admm'].replace("$", "").split(' ')[2]))
    mean_log_admm.append(float(results_dict['test_acc_log_admm'].replace("$", "").split(' ')[0]))
    CI_log_admm.append(float(results_dict['test_acc_log_admm'].replace("$", "").split(' ')[2]))
    mean_ilsr.append(float(results_dict['test_acc_lsr'].replace("$", "").split(' ')[0]))
    CI_ilsr.append(float(results_dict['test_acc_lsr'].replace("$", "").split(' ')[2]))
    mean_newton_on_beta.append(float(results_dict['test_acc_newton_exp_beta'].replace("$", "").split(' ')[0]))
    CI_newton_on_beta.append(float(results_dict['test_acc_newton_exp_beta'].replace("$", "").split(' ')[2]))
mean_lin_admm = np.array(mean_lin_admm)
CI_lin_admm = np.array(CI_lin_admm)
mean_log_admm = np.array(mean_log_admm)
CI_log_admm = np.array(CI_log_admm)
mean_ilsr = np.array(mean_ilsr)
CI_ilsr = np.array(CI_ilsr)
mean_newton_on_beta = np.array(mean_newton_on_beta)
CI_newton_on_beta = np.array(CI_newton_on_beta)
y_label = 'Top-1 Acc.'
save_fig()

###########################
p_range = [10,100,1000,10000]
n_ptest = 1000
M_ptest = 250
xaxis = p_range
x_label = 'p'
######## Conv Time vs p
mean_lin_admm = []
CI_lin_admm = []
mean_log_admm = []
CI_log_admm = []
mean_ilsr = []
CI_ilsr = []
mean_newton_on_beta = []
CI_newton_on_beta = []
for p in p_range:
    save_name = str(rand_iter) + '_' + str(test_fold) + '_n_' + str(n_ptest) + '_p_' + str(p) + '_k_' + str(2) + '_d_' + str(M_ptest) + '_rho_' + str(rho)
    results_dict = metric_and_CI_over_seeds(['test', 'lsr', 'mm', 'lin_admm', 'log_admm', 'theta_newton',
                                             'beta_newton_exp_beta', 'slsqp'], dir, save_name, n_seeds=n_fold,
                                            n_fold=n_fold)  # key:mean, val:CI
    mean_lin_admm.append(float(results_dict['time_cont_lin_admm'].replace("$", "").split(' ')[0]))
    CI_lin_admm.append(float(results_dict['time_cont_lin_admm'].replace("$", "").split(' ')[2]))
    mean_log_admm.append(float(results_dict['time_cont_log_admm'].replace("$", "").split(' ')[0]))
    CI_log_admm.append(float(results_dict['time_cont_log_admm'].replace("$", "").split(' ')[2]))
    mean_ilsr.append(float(results_dict['time_cont_lsr'].replace("$", "").split(' ')[0]))
    CI_ilsr.append(float(results_dict['time_cont_lsr'].replace("$", "").split(' ')[2]))
    mean_newton_on_beta.append(float(results_dict['time_cont_newton_exp_beta'].replace("$", "").split(' ')[0]))
    CI_newton_on_beta.append(float(results_dict['time_cont_newton_exp_beta'].replace("$", "").split(' ')[2]))
mean_lin_admm = np.array(mean_lin_admm)
CI_lin_admm = np.array(CI_lin_admm)
mean_log_admm = np.array(mean_log_admm)
CI_log_admm = np.array(CI_log_admm)
mean_ilsr = np.array(mean_ilsr)
CI_ilsr = np.array(CI_ilsr)
mean_newton_on_beta = np.array(mean_newton_on_beta)
CI_newton_on_beta = np.array(CI_newton_on_beta)
y_label = 'Time'
save_fig()
######## Accuracy vs p
mean_lin_admm = []
CI_lin_admm = []
mean_log_admm = []
CI_log_admm = []
mean_ilsr = []
CI_ilsr = []
mean_newton_on_beta = []
CI_newton_on_beta = []
for p in p_range:
    save_name = str(rand_iter) + '_' + str(test_fold) + '_n_' + str(n_ptest) + '_p_' + str(p) + '_k_' + str(2) + '_d_' + str(M_ptest) + '_rho_' + str(rho)
    results_dict = metric_and_CI_over_seeds(['train', 'lsr', 'mm', 'lin_admm', 'log_admm', 'theta_newton',
                                             'beta_newton_exp_beta', 'slsqp'], dir, save_name, n_seeds=n_fold,
                                            n_fold=n_fold)  # key:mean, val:CI
    mean_lin_admm.append(float(results_dict['test_acc_lin_admm'].replace("$", "").split(' ')[0]))
    CI_lin_admm.append(float(results_dict['test_acc_lin_admm'].replace("$", "").split(' ')[2]))
    mean_log_admm.append(float(results_dict['test_acc_log_admm'].replace("$", "").split(' ')[0]))
    CI_log_admm.append(float(results_dict['test_acc_log_admm'].replace("$", "").split(' ')[2]))
    mean_ilsr.append(float(results_dict['test_acc_lsr'].replace("$", "").split(' ')[0]))
    CI_ilsr.append(float(results_dict['test_acc_lsr'].replace("$", "").split(' ')[2]))
    mean_newton_on_beta.append(float(results_dict['test_acc_newton_exp_beta'].replace("$", "").split(' ')[0]))
    CI_newton_on_beta.append(float(results_dict['test_acc_newton_exp_beta'].replace("$", "").split(' ')[2]))
mean_lin_admm = np.array(mean_lin_admm)
CI_lin_admm = np.array(CI_lin_admm)
mean_log_admm = np.array(mean_log_admm)
CI_log_admm = np.array(CI_log_admm)
mean_ilsr = np.array(mean_ilsr)
CI_ilsr = np.array(CI_ilsr)
mean_newton_on_beta = np.array(mean_newton_on_beta)
CI_newton_on_beta = np.array(CI_newton_on_beta)
y_label = 'Top-1 Acc.'
save_fig()

###########################
M_range = [10,100,1000,10000,100000]
n_Mtest = 1000
p_Mtest = 100
xaxis = M_range
x_label = 'M'
######## Conv Time vs M
mean_lin_admm = []
CI_lin_admm = []
mean_log_admm = []
CI_log_admm = []
mean_ilsr = []
CI_ilsr = []
mean_newton_on_beta = []
CI_newton_on_beta = []
for M in M_range:
    save_name = str(rand_iter) + '_' + str(test_fold) + '_n_' + str(n_Mtest) + '_p_' + str(p_Mtest) + '_k_' + str(2) + '_d_' + str(M) + '_rho_' + str(rho)
    results_dict = metric_and_CI_over_seeds(['test', 'lsr', 'mm', 'lin_admm', 'log_admm', 'theta_newton',
                                             'beta_newton_exp_beta', 'slsqp'], dir, save_name, n_seeds=n_fold,
                                            n_fold=n_fold)  # key:mean, val:CI
    mean_lin_admm.append(float(results_dict['time_cont_lin_admm'].replace("$", "").split(' ')[0]))
    CI_lin_admm.append(float(results_dict['time_cont_lin_admm'].replace("$", "").split(' ')[2]))
    mean_log_admm.append(float(results_dict['time_cont_log_admm'].replace("$", "").split(' ')[0]))
    CI_log_admm.append(float(results_dict['time_cont_log_admm'].replace("$", "").split(' ')[2]))
    mean_ilsr.append(float(results_dict['time_cont_lsr'].replace("$", "").split(' ')[0]))
    CI_ilsr.append(float(results_dict['time_cont_lsr'].replace("$", "").split(' ')[2]))
    mean_newton_on_beta.append(float(results_dict['time_cont_newton_exp_beta'].replace("$", "").split(' ')[0]))
    CI_newton_on_beta.append(float(results_dict['time_cont_newton_exp_beta'].replace("$", "").split(' ')[2]))
mean_lin_admm = np.array(mean_lin_admm)
CI_lin_admm = np.array(CI_lin_admm)
mean_log_admm = np.array(mean_log_admm)
CI_log_admm = np.array(CI_log_admm)
mean_ilsr = np.array(mean_ilsr)
CI_ilsr = np.array(CI_ilsr)
mean_newton_on_beta = np.array(mean_newton_on_beta)
CI_newton_on_beta = np.array(CI_newton_on_beta)
y_label = 'Time'
save_fig()
######## Accuracy vs M
mean_lin_admm = []
CI_lin_admm = []
mean_log_admm = []
CI_log_admm = []
mean_ilsr = []
CI_ilsr = []
mean_newton_on_beta = []
CI_newton_on_beta = []
for M in M_range:
    save_name = str(rand_iter) + '_' + str(test_fold) + '_n_' + str(n_Mtest) + '_p_' + str(p_Mtest) + '_k_' + str(2) + '_d_' + str(M) + '_rho_' + str(rho)
    results_dict = metric_and_CI_over_seeds(['test', 'lsr', 'mm', 'lin_admm', 'log_admm', 'theta_newton',
                                             'beta_newton_exp_beta', 'slsqp'], dir, save_name, n_seeds=n_fold,
                                            n_fold=n_fold)  # key:mean, val:CI
    mean_lin_admm.append(float(results_dict['test_acc_lin_admm'].replace("$", "").split(' ')[0]))
    CI_lin_admm.append(float(results_dict['test_acc_lin_admm'].replace("$", "").split(' ')[2]))
    mean_log_admm.append(float(results_dict['test_acc_log_admm'].replace("$", "").split(' ')[0]))
    CI_log_admm.append(float(results_dict['test_acc_log_admm'].replace("$", "").split(' ')[2]))
    mean_ilsr.append(float(results_dict['test_acc_lsr'].replace("$", "").split(' ')[0]))
    CI_ilsr.append(float(results_dict['test_acc_lsr'].replace("$", "").split(' ')[2]))
    mean_newton_on_beta.append(float(results_dict['test_acc_newton_exp_beta'].replace("$", "").split(' ')[0]))
    CI_newton_on_beta.append(float(results_dict['test_acc_newton_exp_beta'].replace("$", "").split(' ')[2]))
mean_lin_admm = np.array(mean_lin_admm)
CI_lin_admm = np.array(CI_lin_admm)
mean_log_admm = np.array(mean_log_admm)
CI_log_admm = np.array(CI_log_admm)
mean_ilsr = np.array(mean_ilsr)
CI_ilsr = np.array(CI_ilsr)
mean_newton_on_beta = np.array(mean_newton_on_beta)
CI_newton_on_beta = np.array(CI_newton_on_beta)
y_label = 'Top-1 Acc.'
save_fig()
