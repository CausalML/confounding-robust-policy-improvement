
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx
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
module_path = os.path.abspath(os.path.join('/Users/az/Box Sync/unconfoundedness'))
if module_path not in sys.path:
    sys.path.append(module_path)
from scripts_running import *
from unconfoundedness_fns import *
# from evals import *
from greedy_partitioning import *
from subgrad import *
from scipy.stats import norm
from scipy.stats import multivariate_normal
from copy import deepcopy

d = 5  # dimension of x
n = 800;
# parameters
rho = np.asarray([1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0, 1 / np.sqrt(3)])  # normalize to unit 0.5
rho = rho / (np.dot(rho, rho) * 2)

beta_cons = 2.5
beta_x = np.asarray([0, .5, -0.5, 0, 0, 0])
beta_x_T = np.asarray([-1.5, 1, -1.5, 1., 0.5, 0])
beta_T = np.asarray([0, .75, -.5, 0, -1, 0, 0])
beta_T_conf = np.asarray([0, .75, -.5, 0, -1, 0])
# beta_T = np.asarray([-1, 0,0, -.5, 1,0.5,1.5])
# beta_T_conf = np.asarray([-1, 0,0, -.5, 1,0.5 ])
mu_x = np.asarray([-1, .5, -1, 0, -1]);

alpha = -2
w = 1.5

def return_CATE_optimal_assignment(x, u):
    n = x.shape[0]
    risk_T_1 = real_risk_(np.ones(n), x, u)
    risk_T_0 = real_risk_(np.zeros(n), x, u)
    opt_T = [1 if risk_T_1[k] < risk_T_0[k] else 0 for k in range(n)]
    return opt_T


# generate propensity model
def REAL_PROP_LOG(x, u):
    Gamma = 1.5
    print x.shape
    nominal_ = logistic_pol_asgn(beta_T_conf, x)
    #     return nominal_
    # set u = I[ Y(T)\mid x > Y(-T) \mid x ]
    a_bnd, b_bnd = get_bnds(nominal_, Gamma)
    q_lo = 1 / b_bnd;
    q_hi = 1 / a_bnd
    opt_T = return_CATE_optimal_assignment(x, u)
    q_real = np.asarray([q_hi[i] if opt_T[i] == 1 else q_lo[i] for i in range(len(u))])

    return q_real

def real_risk(T, beta_cons, beta_x, beta_x_T, x, u):
    '''
    takes as input integral T
    '''
    n = len(T);
    risk = np.zeros(n)
    for i in range(len(T)):
        risk[i] = T[i] * beta_cons + np.dot(beta_x.T, x[i, :]) + np.dot(beta_x_T.T, x[i, :] * T[i]) + alpha * (u[i]) * (
        (2 * T[i] - 1)) + w * (u[i])
    return risk


def real_risk_(T, x, u):
    return real_risk(T, beta_cons, beta_x, beta_x_T, x, u)

def real_risk_T_integer(T, x, u):
    T = get_sgn_0_1(T)
    return real_risk(T, beta_cons, beta_x, beta_x_T, x, u)


'''
takes in prob_1, n x m array of treatment assignment probabilities 
compute 
\sum_t 1/n \sum_i Y(t)\pi_i(t) 
'''
def real_risk_prob_oracle(prob_t, x, u):
    n_ts = prob_t.shape[1]
    risk = 0
    for t in range(n_ts):
        risk += np.mean( np.multiply(prob_t[:,t], real_risk_(t*np.ones(x.shape[0]), x, u) )  )
    return risk
#     return prob_1 * real_risk_T_integer(np.ones(n), x, u) + (1 - prob_1) * real_risk_(np.zeros(n), x, u)




def real_risk_prob(prob_1, x, u):
    n = len(u)
    prob_1 = np.asarray(prob_1)
    return prob_1 * real_risk_(np.ones(n), x, u) + (1 - prob_1) * real_risk_(np.zeros(n), x, u)

def generate_log_data(mu_x, n, beta_cons, beta_x, beta_x_T):
    # generate n datapoints from the same multivariate normal distribution
    d = len(mu_x)
    u = (np.random.rand(n) > 0.5)  # needs to be Rademacher
    #     u = np.asarray([u[i] if u[i] == 1 else -1 for i in range(len(u))])
    #     u = np.ones(n)
    x = np.zeros([n, d])
    for i in range(n):
        x[i, :] = np.random.multivariate_normal(mean=mu_x * (2 * u[i] - 1), cov=np.eye(d))
    x_ = np.hstack([x, np.ones([n, 1])])
    # generate propensities
    true_Q = REAL_PROP_LOG(x_, u)
    T = np.array(np.random.uniform(size=n) < true_Q).astype(int).flatten()
    T = T.reshape([n, 1]);
    T_sgned = np.asarray([1 if T[i] == 1 else -1 for i in range(n)]).flatten()
    clf = LogisticRegression();
    clf.fit(x, T)
    propensities = clf.predict_proba(x)
    nominal_propensities_pos = propensities[:, 1]

    nominal_propensities_pos = logistic_pol_asgn(beta_T_conf, x_)

    q0 = np.asarray([nominal_propensities_pos[i] if T[i] == 1 else 1 - nominal_propensities_pos[i] for i in range(n)])
    true_Q_obs = np.asarray([true_Q[i] if T[i] == 1 else 1 - true_Q[i] for i in range(n)])
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = T[i] * beta_cons + np.dot(beta_x.T, x_[i, :]) + np.dot(beta_x_T.T, x_[i, :] * T[i]) + alpha * (u[i]) * (
                    2 * T[i] - 1) + w * u[i]  # + np.dot(beta_x_T.T, x_[i,:] - [-1,1] ) #+ np.dot(beta_x_high_freq.T, np.sin(x[i,0:HIGH_FREQ_N]*FREQ)*T[i])
    # add random noise
    T = T.flatten()
    Y += 2 * np.random.randn(n)  # + np.random.randn()*np.asarray(T)*2 ;
    return [x_, u, T, Y, true_Q_obs, q0]


# generate propensity model
def REAL_PROP_LOG(x, u):
    Gamma = 1.5
    print x.shape
    nominal_ = logistic_pol_asgn(beta_T_conf, x)
    #     return nominal_
    # set u = I[ Y(T)\mid x > Y(-T) \mid x ]
    a_bnd, b_bnd = get_bnds(nominal_, Gamma)
    q_lo = 1 / b_bnd;
    q_hi = 1 / a_bnd
    opt_T = return_CATE_optimal_assignment(x, u)
    q_real = np.asarray([q_hi[i] if opt_T[i] == 1 else q_lo[i] for i in range(len(u))])

    return q_real


def real_risk_mt(x_, u, T, alpha, beta_tilde, beta_x_T,  eta_tilde, eta):
    n = len(T); risk = np.zeros(n)
    for i in range(len(T)):
        risk[i] = alpha[T[i]] + np.dot(beta_tilde.T, x_[i, :]) + np.dot(beta_x_T[:,T[i]].T, x_[i, :]) + eta[T[i]]*(u[i]) + eta_tilde*u[i]
    return risk

def real_risk_mt_(x_, u, T):
    return real_risk_mt(x_, u, T,*outcome_dgp_params)

'''
takes in prob_1, n x m array of treatment assignment probabilities 
compute 
\sum_t 1/n \sum_i Y(t)\pi_i(t) 
'''
def real_risk_prob_oracle_mt(prob_t, x, u, *outcome_dgp_params):
    n_ts = prob_t.shape[1]
    risk = 0
    for t in range(n_ts):
        risk += np.mean( np.multiply(prob_t[:,t], real_risk_mt_(x, u, t*np.ones(n).astype(int)) )  )
    return risk
#     return prob_1 * real_risk_T_integer(np.ones(n), x, u) + (1 - prob_1) * real_risk_(np.zeros(n), x, u)


def return_CATE_optimal_assignment_mt(x, u, n_ts, *outcome_dgp_params):
    n = x.shape[0]
    risk = np.zeros([n,n_ts])
    for k in range(n_ts):
        risk[:,k] = real_risk_mt(x, u, k*np.ones(n).astype(int), *outcome_dgp_params)
    opt_T = [np.argmin(risk[i,:]) for i in range(n)]
    return opt_T

''' real propensity that generates treatment assignments for multiple treatments 
'''
def real_propensity_mt(x, u, beta_T_conf_prop, log_gamma, *outcome_dgp_params ):
    nominal_ = logistic_pol_asgn_mt(beta_T_conf_prop, x)
    # set u = I[ Y(T)\mid x > Y(-T) \mid x ]
    opt_T = return_CATE_optimal_assignment_mt(x, u, beta_T_conf_prop.shape[1], *outcome_dgp_params)
    n_ts = beta_T_conf_prop.shape[1]
    q_lo = np.zeros([x.shape[0], n_ts]); q_hi = np.zeros([x.shape[0], n_ts])
    for k in range(n_ts):
        a_bnd, b_bnd = get_bnds(nominal_[:,k], log_gamma)
        q_lo[:,k] = 1 / b_bnd;
        q_hi[:,k] = 1 / a_bnd
    q_real = deepcopy(nominal_)
    for i in range(x.shape[0]):
        if opt_T[i] == 1:
            q_real[i,1] = q_hi[i, 1 ]
            q_real[i,2] = nominal_[i,2]
            q_real[i,0] = 1 - q_real[i,1] - q_real[i,2]
        # elif opt_T[i] == 0:
        #     q_real[i,0] = q_lo[i, 0 ]
        #     q_real[i,2] = nominal_[i,2]
        #     q_real[i,1] = 1 - q_real[i,0] - q_real[i,2]
        else:
            q_real[i,:] = nominal_[i,:]
    # clip and renormalize
    q_real = np.clip(q_real, 0.01, 0.99)
    for i in range(x.shape[0]):
        q_real[i,:] = q_real[i,:] / sum(q_real[i,:])
    return q_real


'''
# take in a vector of
beta_cons: 
beta_tilde: \tilde{\beta} (confounding interaction with x) 
beta_x_T: [d x k] array of coefficient vectors, one for each treatment

'''
def generate_log_data_mt(mu_x, n, beta_T_conf_prop, n_ts, log_gamma, alpha, beta_tilde, beta_x_T,  eta_tilde, eta):
    # generate n datapoints from the same multivariate normal distribution
    outcome_dgp_params = [alpha, beta_tilde, beta_x_T,  eta_tilde, eta ]
    d = len(mu_x);
    u = (np.random.rand(n) > 0.5)  # Bernoulli noise realization;
    x = np.zeros([n, d])
    for i in range(n):
        x[i, :] = np.random.uniform(low=np.ones(d)*-3,high = np.ones(d)*3)

        # x[i, :] = np.random.multivariate_normal(mean=mu_x * (2 * u[i] - 1), cov=np.eye(d))
    x_ = np.hstack([x, np.ones([n, 1])])
    # generate propensities
    true_Q = real_propensity_mt(x_, u, beta_T_conf_prop, log_gamma, *outcome_dgp_params ) # in n x k for multiple treatments
    T = np.asarray([np.random.choice( range(n_ts), p = true_Q[i,:] ) for i in range(n) ]).astype(int).flatten()
    q0 = logistic_pol_asgn_mt_obsA(beta_T_conf_prop, x_, T)
    # q0 = np.asarray([nominal_propensities_pos[i,T[i]] for i in range(n)])
    true_Q_obs = np.asarray([true_Q[i,T[i]] for i in range(n)])
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = alpha[T[i]] + np.dot(beta_tilde.T, x_[i, :]) + np.dot(beta_x_T[:,T[i]].T, x_[i, :]) + eta[T[i]]*(u[i]) + eta_tilde*u[i]
    # add random noise
    Y += np.random.randn(n)  # + np.random.randn()*np.asarray(T)*2 ;
    return [x_, u, T, Y, true_Q_obs, q0]
