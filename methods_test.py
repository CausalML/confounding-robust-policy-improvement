import matplotlib.pyplot as plt
import numpy as np
from gurobipy import *
from scipy.optimize import minimize
import datetime as datetime
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import collections as matcoll
from sklearn import svm
from methods import *
import os
import sys
from unconfoundedness_fns import *
from subgrad import *
from scipy.stats import norm
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from data_scenarios import *

def real_risk_prob(prob_1, x, u):
    n = len(u)
    prob_1 = np.asarray(prob_1)
    return prob_1 * real_risk_(np.ones(n), x, u) + (1 - prob_1) * real_risk_(np.zeros(n), x, u)

N_REPS = int(sys.argv[1])
arr = sys.argv[2].split(',')
GAMS = [float(gam) for gam in arr]
print(GAMS), 'gammas'

ENN = 4000
d = 5  # dimension of x
th_ctrl = np.zeros(d+1)
th_ctrl[-1] = -1000

beta_cons = 2.5
beta_x = np.asarray([0, .5, -0.5, 0, 0, 0])
beta_x_T = np.asarray([-1.5,1,-1.5,1.,0.5,0])
beta_T = np.asarray([0, .75, -.5,0,-1,0, 0])
beta_T_conf = np.asarray([0, .75, -.5,0,-1,0])
mu_x = np.asarray([-1,.5,-1,0,-1]);

dgp_params = {'mu_x':mu_x, 'n':ENN, 'beta_cons':beta_cons, 'beta_x':beta_x, 'beta_x_T':beta_x_T}
save_params = {'stump':'sharp-test-synthetic/', 'exp_name': 'testing'}


opt_config_robust = {'N_RST': 15, 'GRAD_': get_implicit_grad_centered, 'WGHTS_': opt_wrapper,
                     'GRAD_CTR': get_implicit_grad_centered, 'POL_PROB_1': logistic_pol_asgn,
                     'POL_GRAD': qk_dpi_dtheta, 'DEFAULT_POL':th_ctrl,
                     'BASELINE_POL': ctrl_p_1, 'P_1': ctrl_p_1, 'averaging': True, 'give_initial': True,
                     'sharp': True}

robust_opt_params = {'optimizer': opt_w_restarts, 'pol_opt': 'ogd',
                     'unc_set_type': 'interval', 'opt_params': opt_config_robust,
                     'BASELINE_POL': th_ctrl, 'type': 'logistic-interval'}

fdiv_config = {'N_RST':15,'DEFAULT_POL':th_ctrl,
               'rho':0.5, 'WGHTS_':get_general_interval_wghts_algo_centered_TV_prob, 'GRAD_':get_implicit_grad_centered,
               'GRAD_CTR':get_implicit_grad_centered, 'POL_PROB_1':logistic_pol_asgn,'POL_GRAD':qk_dpi_dtheta,
        'BASELINE_POL': ctrl_p_1,'eta_0':0.5, 'averaging':True, 'give_initial':True, 'sharp':True}

fdiv_robust_opt_params = { 'optimizer':opt_w_restarts, 'pol_opt':'ogd',
                     'unc_set_type':'L1-budget', 'opt_params':fdiv_config,
                    'BASELINE_POL':th_ctrl, 'type':'logistic-L1-budget-0.5'}

fdiv_config_025 = {'N_RST':15,'DEFAULT_POL':th_ctrl,
               'rho':0.25, 'WGHTS_':get_general_interval_wghts_algo_centered_TV_prob, 'GRAD_':get_implicit_grad_centered,
               'GRAD_CTR':get_implicit_grad_centered, 'POL_PROB_1':logistic_pol_asgn,'POL_GRAD':qk_dpi_dtheta,
        'BASELINE_POL': ctrl_p_1,'eta_0':0.5, 'averaging':True, 'give_initial':True, 'sharp':True}

fdiv_robust_opt_params_025 = { 'optimizer':opt_w_restarts, 'pol_opt':'ogd',
                     'unc_set_type':'L1-budget', 'opt_params':fdiv_config_025,
                    'BASELINE_POL':th_ctrl, 'type':'logistic-L1-budget-0.25'}

#########
## IPW and tree parameters
opt_config_ipw = {'N_RST':15, 'N_RNDS':50, 'POL_PROB_1':logistic_pol_asgn, 'eta_0':1, 'averaging':True,'DEFAULT_POL':th_ctrl}

ipw_opt_params = { 'optimizer':opt_w_restarts_vanilla_ipw, 'pol_opt':'IPW',
                     'unc_set_type':'interval', 'opt_params':opt_config_ipw,
                    'BASELINE_POL':th_ctrl, 'type':'IPW'}


methods = ['ogd-interval', 'fdiv-0.5', 'fdiv-0.25']
method_params = [robust_opt_params, fdiv_robust_opt_params, fdiv_robust_opt_params_025]


def gen_data_run_for_gamma_for_joblib_mt(dgp_params, GAMS, real_risk_prob, method_params, ind_rep, gen_data=True,
                                      save=False, save_params=[], already_gen_data=[]):
    print ind_rep
    if gen_data:
        [x_full, u, T_, Y_, true_Q_, q0] = generate_log_data(**dgp_params)
    else:
        [x_full, u, T_, Y_, true_Q_, q0] = already_gen_data
    np.random.seed(ind_rep)
    random.seed(ind_rep)
    train_ind, test_ind = model_selection.train_test_split(range(len(Y_)), test_size=0.9)
    test_data = {'x_test': x_full[test_ind, :], 't_test': T_[test_ind], 'y_test': Y_[test_ind], 'u_test': u[test_ind]}
    eval_conf = {'eval': True, 'eval_type': 'true_dgp', 'eval_data': test_data, 'oracle_risk': real_risk_prob_oracle}
    ConfRobPols = [ConfoundingRobustPolicy(baseline_pol=ctrl_p_1_mt, save_params=save_params, save = True, verbose=True, treatment_n = 'multiple') for method in
                   method_params]
    for ind, method_param in enumerate(method_params):
        ConfRobPols[ind].fit(x_full[train_ind, :], T_[train_ind], Y_[train_ind], q0[train_ind], GAMS, method_param,
                             eval_conf=eval_conf)
        del ConfRobPols[ind].x
        del ConfRobPols[ind].t
        del ConfRobPols[ind].y
        del ConfRobPols[ind].eval_data

    return ConfRobPols


res = Parallel(n_jobs=-1, verbose=40)(
    delayed(gen_data_run_for_gamma_for_joblib_mt)(dgp_params, GAMS, real_risk_prob, method_params, i, gen_data=True,
                                               save=True, save_params=save_params) for i in range(N_REPS))

#     delayed(gen_data_run_for_gamma_for_joblib)(dgp_params, GAMS, real_risk_prob, method_params, i, gen_data=True,
#                                                save=True, save_params=save_params) for i in range(N_REPS))

pickle.dump(res, open('res-'+save_params['exp_name'] + '.pkl', 'wb'))
RISKS = [[ res[ind][meth].RISKS for ind in range(N_REPS) ] for meth in range(len(method_params)) ]
pickle.dump(RISKS, open('risks_joblib_test_opt_tree_mt_50reps.pkl', 'wb') )
