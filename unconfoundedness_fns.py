"""
@author: Angela Zhou
"""
import numpy as np
from random import sample
import math
import cvxpy as cvx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# from matplotlib import collections as matcoll
from sklearn import svm
from scipy.integrate import quad
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from scipy import integrate
# from sympy import mpmath as mp




'''
Helper functions for estimating propensities
'''
def estimate_prop(x, T, predict_x, predict_T):
    clf_dropped = LogisticRegression()
    clf_dropped.fit(x, T)
    est_prop = clf_dropped.predict_proba(predict_x)
    est_Q = np.asarray( [est_prop[k,1] if predict_T[k] == 1 else est_prop[k,0] for k in range(len(predict_T))] )
    return [est_Q, clf_dropped]

def get_prop(clf, x, T):
    est_prop = clf_dropped.predict_proba(x)
    est_Q = np.asarray( [est_prop[k,1] if T[k] == 1 else est_prop[k,0] for k in range(len(T))] )
    return est_Q



# get indicator vector from signed vector
def get_0_1_sgn(vec):
    n = len(vec)
    return np.asarray([1 if vec[i] == 1 else 0 for i in range(n) ]).flatten()
# get signed vector from indicator vector
def get_sgn_0_1(vec):
    n = len(vec)
    return np.asarray([1 if vec[i] == 1 else -1 for i in range(n) ]).flatten()
'''
performs policy match indicator function; returns 1 or 0
input: signed treatments, signed policy assignment
'''
def pol_match(T_sgned, pol):
    sgn_match = np.multiply(T_sgned, pol )
    return get_0_1_sgn(sgn_match)





''' faster sorting routines
'''
''' Get general weights by sorting for the uncentered, smoothed estimator
'''
def get_general_interval_wghts_algo_uncentered_smoothed(init_policy, **params):
    T_obs = params['T'].astype(int) ; Y = params['Y'].flatten(); n = params['n']; x = params['x'];
    n = params['n']; weights = np.zeros(n); a = params['a']; b = params['b']
    T_sgned=get_sgn_0_1(T_obs)
    probs_pi_T = params['pi_pol'](T_sgned, init_policy, x)
    a_mod = np.multiply(a, probs_pi_T); b_mod = np.multiply(b, probs_pi_T)
    # Sort all Y!
    plus_inds = np.argsort(Y);
    Y_plus_std = Y[plus_inds] # Sorted Y
    n_plus = n; vals = np.zeros(n_plus+1)
    prev_val = -np.inf; k=0; val = np.inf
    ## !!!! You can speed up by binary search
    while (val > prev_val) and (k < n_plus+1):
        denom = 0; num = 0; val = prev_val
#     for i in range(len(vals)):
        num = sum(np.multiply(a_mod[plus_inds[0:k]], Y_plus_std[0:k])) + sum(np.multiply(b_mod[plus_inds[k:]], Y_plus_std[k:]))
        denom = sum(a_mod[plus_inds[0:k]])+sum(b_mod[plus_inds[k:]])
#         for k in range(n_plus):
#             if k<i: # threshold on order statistics
#                 num += a_mod[plus_inds][k]*Y_plus_std[k]
#                 denom += a_mod[plus_inds][k]
#             else:
#                 num += b_mod[plus_inds][k]*Y_plus_std[k]
#                 denom += b_mod[plus_inds][k]
#         vals[i] = vals[i] / denom
        val = num / denom; k+=1;
    lda_opt = val; k_star = k
#     lda_opt = np.max(vals); k_star = np.argmax(vals) #kstar follows python indexing
    weights[plus_inds[0:k_star]] = a_mod[plus_inds[0:k_star]]
    weights[plus_inds[k_star:]] = b_mod[plus_inds[k_star:]]
    return [weights, sum(weights)]


'''
copypasted from ipynb
'''
# def get_optimal_interval_wghts_CVX(init_policy, **params):
#     T_obs = params['T'].astype(int) ; Y = params['Y']; n = params['n'];
#     x = params['x']; a = params['a']; b = params['b']
#     W = cvx.Variable(n); t = cvx.Variable(); hinge = params['hinge']
#     constraints = [ np.multiply(a, hinge) * t <= W, W <= np.multiply(b, hinge) * t  ]
#     constraints += [ sum(W) == 1, t > 0, W >= 0 ]
#     prim_obj =  (  Y.T * W ) #+ LAMBDA*(cvx.norm(opt_theta)+cvx.norm(opt_theta_0) + cvx.norm(opt_theta_1))
#     obj = cvx.Maximize(prim_obj) # -1* minimize negative rewards = max rewards
#     prob = cvx.Problem(obj, constraints) # since we add a convex regularizer
#     prob.solve()
#     return [ W.value, t.value, prob.value ]

'''
solves for optimal weights for the problem with \sum W_i (pi Y_i)
returns w, unnormalized weights
'''
def get_general_interval_wghts_pol(init_policy, **params):
    n = params['n']; weights = np.zeros(n)
    T_sgned = np.asarray([ 1 if params['T'][i] == 1 else -1 for i in range(n)]).flatten()
    policy_x = params['pi_pol'](T_sgned, init_policy, params['x']).flatten()
    params['hinge'] = policy_x
    [W, t, val] = get_optimal_interval_wghts_CVX(init_policy, **params) # W is normalized weights
    weights = (W.flatten()/t).T
    return [weights,t]

'''
evaluate loss function from parametric problem (uncentered), under probabilistic policy assumption
#! weights are weights multiplied separately by probabilities
compute the loss as weights^T * (\pi Y)
'''
def normalized_parametric_loss_all(pol_theta, *args):
    params = dict(args[0]); x = params['x']; C = params['C']; sign = params['sign'];
    n = params['n']; q = params['q']; Y = params['Y'];
    T = params['T']; a = params['a']; b = params['b']; pi = params['pi_pol'];
    T_sgned = np.asarray([ 1 if T[i] == 1 else -1 for i in range(n)]).flatten();
    W = params['weights']; opt_t = params['opt_t']; d = len(pol_theta);
    policy_x = pi(T_sgned, pol_theta, x).reshape([n,1]); #policy_x_prime = pi_prime(pol_theta, x)
    # already accounted for probability
    loss_val = np.dot(params['weights'].reshape([n,1]).T, Y)
#     loss_val = params['weights'].reshape([n,1]).T * np.multiply(policy_x, Y)
    return sign*( loss_val + C*np.linalg.norm(pol_theta,2))


#! This is function returning Pr[ \pi(x) = T_sgned]
def logistic_pol(T_sgned, theta, x):
    n = len(T_sgned); theta = theta.flatten()
    if len(theta) == 1:
        pol_match = np.multiply(T_sgned, np.multiply(x, theta).flatten())
    else:
        pol_match = np.multiply(T_sgned, np.dot(x, theta).flatten())
    LOGIT_TERM_POS = np.ones(n) / ( np.ones(n) + np.exp( -pol_match ))
    return LOGIT_TERM_POS


'''
performs policy match indicator function; returns 1 or 0
input: signed treatments, signed policy assignment
'''
def pol_match(T_sgned, pol):
    sgn_match = np.multiply(T_sgned, pol )
    return get_0_1_sgn(sgn_match)


''' Get general weights by sorting for the uncentered, smoothed estimator
'''
def get_general_interval_wghts_algo_uncentered_smoothed(init_policy, **params):
    T_obs = params['T'].astype(int) ; Y = params['Y'].flatten(); n = params['n']; x = params['x'];
    n = params['n']; weights = np.zeros(n); a = params['a']; b = params['b']
    T_sgned=get_sgn_0_1(T_obs)
    probs_pi_T = params['pi_pol'](T_sgned, init_policy, x)
    a_mod = np.multiply(a, probs_pi_T); b_mod = np.multiply(b, probs_pi_T)
    # Sort all Y!
    plus_inds = np.argsort(Y);
    Y_plus_std = Y[plus_inds] # Sorted Y
    n_plus = n; vals = np.zeros(n_plus+1)
    prev_val = -np.inf; k=1; val = sum(np.multiply(b_mod[plus_inds], Y_plus_std)) /  sum(b_mod[plus_inds])
    ## !!!! You can speed up by binary search
    while (val > prev_val) and (k < n_plus+1):
        denom = 0; num = 0; prev_val = val
        num = sum(np.multiply(a_mod[plus_inds[0:k]], Y_plus_std[0:k])) + sum(np.multiply(b_mod[plus_inds[k:]], Y_plus_std[k:]))
        denom = sum(a_mod[plus_inds[0:k]])+sum(b_mod[plus_inds[k:]])
        val = num / denom; k+=1;
    lda_opt = val; k_star = k-1
    weights[plus_inds[0:k_star]] = a_mod[plus_inds[0:k_star]]
    weights[plus_inds[k_star:]] = b_mod[plus_inds[k_star:]]
    return [weights, sum(weights)]


''' return Pr[ \pi(x)=T ]
'''
def logistic_pol_match_obs(T_sgned, theta, x):
    n = len(T_sgned); pol_match = np.multiply(T_sgned, np.dot(x, theta).flatten())
    LOGIT_TERM_POS = np.ones(n) / ( np.ones(n) + np.exp( -pol_match ))
    return LOGIT_TERM_POS
''' return Pr[ \pi(x)=1 ]
'''
def logistic_pol_asgn(theta, x):
    theta = theta.flatten()
    n = x.shape[0]
    if len(theta) == 1:
        logit = np.multiply(x, theta).flatten()
    else:
        logit = np.dot(x, theta).flatten()
    LOGIT_TERM_POS = np.ones(n) / ( np.ones(n) + np.exp( -logit ))
    return LOGIT_TERM_POS

''' multinomial logistic: return all probabilities 
We index theta as a d x K array 
x is n x d
output is n x K 
'''
def logistic_pol_asgn_mt(theta, x):
    n=x.shape[0]; d = x.shape[1]
    k = theta.shape[1]
    logit = np.zeros([n,k]); LOGIT_TERM_POS = np.zeros([n,k])
    if d == 1:
        for a in range(k):
            logit[:,a] = np.multiply(x, theta[:,a])
    else:
        for a in range(k):
            logit[:,a] = np.dot(x, theta[:,a])
    for i in range(n):
        # compute probability of observed action
        LOGIT_TERM_POS[i,:] = np.asarray([np.exp(logit[i,k_]) / ( np.sum( np.exp(logit[i,:]) ) ) for k_ in range(k) ]) # sum over classes
    return LOGIT_TERM_POS # output is nxk

''' multinomial logistic: return all probabilities 
We index theta as a d x K array 
x is n x d
output is n x K 
(overloaded arguments just in case) 
'''
def logistic_pol_asgn_mt(theta, x, t_test=0):
    n=x.shape[0]; d = x.shape[1]
    k = theta.shape[1]
    logit = np.zeros([n,k]); LOGIT_TERM_POS = np.zeros([n,k])
    if d == 1:
        for a in range(k):
            logit[:,a] = np.multiply(x, theta[:,a])
    else:
        for a in range(k):
            logit[:,a] = np.dot(x, theta[:,a])
    for i in range(n):
        # compute probability of observed action
        LOGIT_TERM_POS[i,:] = np.asarray([np.exp(logit[i,k_]) / ( np.sum( np.exp(logit[i,:]) ) ) for k_ in range(k) ]) # sum over classes
        if np.isinf(np.exp(logit[i,k_])):
            LOGIT_TERM_POS[i] = 1 # overflow fix
    return LOGIT_TERM_POS # output is nxk


''' multinomial logistic: return probability of assigning observed treatment
We index theta as a d x K array 
x is n x d 
Output is filtered based on the observed treatment pattern 
'''
def logistic_pol_asgn_mt_obsA(theta, x, t01):
    n = x.shape[0]
    d = x.shape[1];
    k = theta.shape[1] # n treatments
    logit = np.zeros([n,k])
    LOGIT_TERM_POS = np.zeros(n)
    if d == 1:
        for a in range(k):
            logit[:,a] = np.multiply(x, theta[:,a])
    else:
        for a in range(k):
            logit[:,a] = np.dot(x, theta[:,a])
    # numerically stable version
    amax = max(logit)
    for i in range(n):
        # compute probability of observed action
        LOGIT_TERM_POS[i] = np.exp(logit[i,t01[i]]) / ( np.sum( np.exp(logit[i,:]) ) ) # sum over classes
        if np.isinf(np.exp(logit[i,t01[i]])):
            LOGIT_TERM_POS[i] = 1 # overflow fix
        if np.isnan(LOGIT_TERM_POS[i]):
            print 'num',np.exp(logit[i,t01[i]])
            print 'denom',np.sum( np.exp(logit[i,:]) )
    # if sum(np.isnan(LOGIT_TERM_POS))>0:
    #     print 'nan theta',theta
    #     return LOGIT_TERM_POS
    return LOGIT_TERM_POS # output is nx1


def find_opt_weights_short_val(a_,b_,Y):
    [lda_opt, weights, s_wghts] = find_opt_weights_short(a_, b_, Y)
    return lda_opt

'''get odds ratio bounds on estimated propensities (est_Q) given sensitivity level Gamma
'''
def get_bnds(est_Q,LogGamma):
    n = len(est_Q)
    p_hi = np.multiply(np.exp(LogGamma), est_Q ) / (np.ones(n) - est_Q + np.multiply(np.exp(LogGamma), est_Q ))
    p_lo = np.multiply(np.exp(-LogGamma), est_Q ) / (np.ones(n) - est_Q + np.multiply(np.exp(-LogGamma), est_Q ))
    assert (p_lo < p_hi).all()
    a_bnd = 1/p_hi;
    b_bnd = 1/p_lo
    return [ a_bnd, b_bnd ]



# ''' Given the truncated list of weights, Y (unsorted) and
# return Lambda (problem value), k
# '''
# def find_opt_weights_short(a_, b_, Y, sub_ind=[]):
#     if len(sub_ind)>0:
#         a_=a_[sub_ind]; b_=b_[sub_ind]; Y = Y[sub_ind]
#     sort_inds = np.argsort(Y); a_=a_[sort_inds]; Y = Y[sort_inds]; b_=b_[sort_inds]
#     n_plus = len(Y); weights = np.zeros(n_plus); prev_val = -np.inf; k=1; val = sum(np.multiply(b_, Y)) /sum(b_) ## !!!! You can speed up by binary search
#     while (val > prev_val) and (k < n_plus+1):
#         denom = 0; num = 0; prev_val = val; num = 1.0*sum(np.multiply(a_[0:k], Y[0:k])) + sum(np.multiply(b_[k:], Y[k:]))
#         denom = sum(a_[0:k])+sum(b_[k:]); val = num / denom; k+=1;
#     lda_opt = prev_val; k_star = k-1
#     sort_inds_a = sort_inds[0:k_star]; sort_inds_b = sort_inds[k_star:]
#     weights[sort_inds_a] = a_[0:k_star]; weights[sort_inds_b] = b_[k_star:]
#
#     return [lda_opt, weights, sum(weights)]



''' Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind,
return Lambda (min problem value), weights, sum(weights)
'''
def find_opt_weights_short(Y, a_, b_, sub_ind=[]):
    if len(sub_ind)>0:
        print sub_ind
        a_=a_[sub_ind]; b_=b_[sub_ind]; Y = Y[sub_ind]
    sort_inds = np.lexsort((a_,Y)); a_=a_[sort_inds]; Y = Y[sort_inds]; b_=b_[sort_inds]
    n_plus = len(Y); weights = np.zeros(n_plus); prev_val = -np.inf; k=1; val = sum(np.multiply(b_, Y)) /sum(b_)
    while (val > prev_val) and (k < n_plus+1):
        denom = 0; num = 0; prev_val = val; num = 1.0*sum(np.multiply(a_[0:k], Y[0:k])) + sum(np.multiply(b_[k:], Y[k:]))
        denom = sum(a_[0:k])+sum(b_[k:]); val = num / denom; k+=1;
    lda_opt = prev_val; k_star = k-1
    sort_inds_a = sort_inds[0:k_star]; sort_inds_b = sort_inds[k_star:]
    weights[sort_inds_a] = a_[0:k_star]; weights[sort_inds_b] = b_[k_star:]
    return [lda_opt, weights, sum(weights)]

''' explicit: include plots of all values
'''
def find_opt_weights_plot(Y,a_,b_,sub_ind=[], lexsort = False):
    if len(sub_ind)>0:
        print sub_ind
        a_=a_[sub_ind]; b_=b_[sub_ind]; Y = Y[sub_ind]
    if lexsort:
        sort_inds = np.lexsort((a_,Y))
    else:
        sort_inds = np.argsort(Y);
    a_=a_[sort_inds]; Y = Y[sort_inds]; b_=b_[sort_inds]
    n_plus = len(Y); weights = np.zeros(n_plus); prev_val = -np.inf; k=1;
    val = np.sum(np.multiply(b_, Y)) /np.sum(b_)
    vals = [ (1.0*np.sum(np.multiply(a_[0:k], Y[0:k])) + np.sum(np.multiply(b_[k:], Y[k:])))/(np.sum(a_[0:k])+np.sum(b_[k:])) for k in range(n_plus) ]
    lda_opt = np.max(vals); k_star = np.argmax(vals)-1
    plt.figure()
    plt.plot(vals)
    plt.figure()
    plt.plot(np.diff(vals))
    sort_inds_a = sort_inds[0:k_star]; sort_inds_b = sort_inds[k_star:]
    weights[sort_inds_a] = a_[0:k_star]; weights[sort_inds_b] = b_[k_star:]
    return [lda_opt, weights, sum(weights)]


def rnd_k_val(k_,a_,b_,Y):
    k= int(np.floor(k_)) # floor or round?
    return (1.0*np.sum(np.multiply(a_[0:k], Y[0:k])) + np.sum(np.multiply(b_[k:], Y[k:])))/(np.sum(a_[0:k])+np.sum(b_[k:]))

''' Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind,
use ternary search to find the optimal value.
Possibly off by one error but it shouldn't matter.
return Lambda (min problem value), weights, sum(weights)
'''
def find_opt_weights_shorter(Y, a_, b_, sub_ind=[]):
    if len(sub_ind)>0:
        a_=a_[sub_ind]; b_=b_[sub_ind]; Y = Y[sub_ind]
    # sort_inds = np.argsort(Y);
    sort_inds = np.lexsort((b_-a_,Y));
    a_=a_[sort_inds]; Y = Y[sort_inds]; b_=b_[sort_inds]
    n_plus = len(Y); weights = np.zeros(n_plus); prev_val = -np.inf; k=1; val = np.sum(np.multiply(b_, Y)) /sum(b_)
    left = 0; right = n_plus-1;keepgoing=True
    while keepgoing:
        #left and right are the current bounds; the maximum is between them
#         print (left,right)
#         print (rnd_k_val(leftThird,a_,b_,Y), rnd_k_val(rightThird,a_,b_,Y))
        if abs(right - left) < 2.1: # separation in index space
            k = np.floor((left + right)/2)
            keepgoing=False
        leftThird = left + (right - left)/3
        rightThird = right - (right - left)/3
        if rnd_k_val(leftThird,a_,b_,Y) < rnd_k_val(rightThird,a_,b_,Y):
            left = leftThird
        else:
            right = rightThird
    k_star=int(k); k=int(k)
    lda_opt = (1.0*np.sum(np.multiply(a_[0:k], Y[0:k])) + np.sum(np.multiply(b_[k:], Y[k:])))/(np.sum(a_[0:k])+np.sum(b_[k:]))
    sort_inds_a = sort_inds[0:k_star]; sort_inds_b = sort_inds[k_star:]
    weights[sort_inds_a] = a_[0:k_star]; weights[sort_inds_b] = b_[k_star:]
    return [lda_opt, weights, np.sum(weights)]

'''Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind,
Lambda (min problem value), weights, sum(weights)
'''
def find_opt_robust_ipw_val(Y, a_, b_,shorter=False):
    if shorter:
        [lda_opt, weights, s_wghts] = find_opt_weights_shorter(Y, a_, b_)
    else:
        [lda_opt, weights, s_wghts] = find_opt_weights_short(Y, a_, b_)
    return lda_opt

'''Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind,
Lambda (max problem value), weights, sum(weights)
'''
def find_opt_robust_ipw_val_min(Y, a_,b_,shorter=False):
    if shorter:
        [lda_opt, weights, s_wghts] = find_opt_weights_shorter(-Y, a_, b_)
    else:
        [lda_opt, weights, s_wghts] = find_opt_weights_short(-Y, a_, b_)
    return -lda_opt


''' functions for computing TV divergence
'''

''' get optimal weights with TV constraint
'''
import gurobipy as gp

''' minimize wrt TV bound
'''
def get_general_interval_wghts_algo_uncentered_smoothed_f_divergence_TV(incumbent_pol, quiet=True, **params):
    T_obs = params['T'].astype(int) ; Y = params['Y'].flatten(); n = params['n']; x = params['x'];
    n = params['n']; a = params['a']; b = params['b']
    gamma = params['gamma']; wm = 1/params['q']
    if params['subind'] == True:
        subinds = params['subinds']
        Y = Y[subinds]; n = len(subinds); T_obs = T_obs[subinds]; x = x[subinds]; a= a[subinds]; b= b[subinds]; wm = wm[subinds]
    # assume estimated propensities are probs of observing T_i
    y = Y; weights = np.zeros(n)
     # nominal propensities
    # smoothing probabilities
    T_sgned=get_sgn_0_1(T_obs)
    probs_pi_T = params['pi_pol'](T_sgned, incumbent_pol, x)
    a_mod = np.multiply(a, probs_pi_T); b_mod = np.multiply(b, probs_pi_T)
    wm = np.multiply(wm, probs_pi_T)
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w = [m.addVar(obj = -yy, lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    d = [m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    m.update()
    m.addConstr(gp.quicksum(w)==1)
    m.addConstr(gp.quicksum(d)<=gamma*t)
    for i in range(len(y)):
        m.addConstr(w[i] <= b_mod[i] * t)
        m.addConstr(w[i] >= a_mod[i] * t)
        m.addConstr(d[i] >=   w[i] - t*wm[i])
        m.addConstr(d[i] >= - w[i] + t*wm[i])
    m.optimize()
    return -m.ObjVal

''' maximize wrt TV bound
'''
def get_general_interval_wghts_algo_uncentered_smoothed_f_divergence_TV_max(incumbent_pol, quiet=True, **params):
    T_obs = params['T'].astype(int) ; Y = params['Y'].flatten(); n = params['n']; x = params['x'];
    n = params['n']; weights = np.zeros(n); a = params['a']; b = params['b']
    gamma = params['gamma']
    # assume estimated propensities are probs of observing T_i
    y = -Y
    wm = 1/params['q'] # nominal propensities
    # smoothing probabilities
    T_sgned=get_sgn_0_1(T_obs)
    probs_pi_T = params['pi_pol'](T_sgned, incumbent_pol, x)
    a_mod = np.multiply(a, probs_pi_T); b_mod = np.multiply(b, probs_pi_T)
    wm = np.multiply(wm, probs_pi_T)
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w = [m.addVar(obj = -yy, lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    d = [m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    m.update()
    m.addConstr(gp.quicksum(w)==1)
    m.addConstr(gp.quicksum(d)<=gamma*t)
    for i in range(len(y)):
        m.addConstr(w[i] <= b_mod[i] * t)
        m.addConstr(w[i] >= a_mod[i] * t)
        m.addConstr(d[i] >=   w[i] - t*wm[i])
        m.addConstr(d[i] >= - w[i] + t*wm[i])
    m.optimize()
    return m.ObjVal

def logistic_pol_asgn(theta, x):
    n = x.shape[0]
    theta = theta.flatten()
    if len(theta) == 1:
        logit = np.multiply(x, theta).flatten()
    else:
        logit = np.dot(x, theta).flatten()
    LOGIT_TERM_POS = np.ones(n) / ( np.ones(n) + np.exp( -logit ))
    return LOGIT_TERM_POS

####
''' Getting subgradients for the robust value function
'''

'''
read in callbacks for derivative of pi given theta and optimal w, t
PI_1, POL_GRAD (returns (p x 1) vector)
take in ** normalized weights W **
'''
def get_implicit_grad_centered(pol_theta, PI_1, POL_GRAD, x, Y, t01, W):
    # if need to get active index set
    # rescaled weights in original
    n = len(W); T_sgned = get_sgn_0_1(t01)

    # dc_dpi = np.diag(Y*T_sgned)
    # policy_x = PI_1(pol_theta, x) # 1 x n
    # dpi_dtheta = POL_GRAD(policy_x, pol_theta, x) # n x p
    # return dc_dpi.dot(dpi_dtheta).T.dot(W)
    constants = np.multiply(Y, np.multiply(T_sgned,W))
    policy_x = PI_1(pol_theta, x) # 1 x n
    dpi_dtheta = POL_GRAD(policy_x, pol_theta, x) # n x p
    if x.ndim > 1:
        return np.multiply(constants[:,np.newaxis], dpi_dtheta).sum(axis=0)
    else:
        return np.sum(np.multiply(constants,dpi_dtheta))
# # Minibatch version:
#     CHUNKN = 20
#     id_batches = np.array_split(range(n),CHUNKN) # list of arrays of indices
#     if x.ndim > 1:
#         grad = np.zeros([x.shape[1], 1])
#     else:
#         grad = 0
#     for batch in id_batches:
#         constants = Y[batch]*T_sgned[batch]*W[batch]
#         policy_x = PI_1(pol_theta, x[batch,]) # 1 x n
#         dpi_dtheta = POL_GRAD(policy_x, pol_theta, x[batch,]) # n x p
#         grad += np.sum(np.multiply(constants[:,np.newaxis], dpi_dtheta)) # should be in px1
#     return grad # no regularization
'''
read in callbacks for derivative of pi given theta and optimal w, t
Y(Pi) - Y(-Pi)
PI_1, POL_GRAD (returns (p x 1) vector)
take in ** normalized weights W **

'''
def get_implicit_grad_centered_anti_pi(pol_theta, PI_1, POL_GRAD, x, Y, t01, W):
    n = len(W); T_sgned = get_sgn_0_1(t01)
    dc_dpi = np.diag(2*Y*T_sgned)
    policy_x = PI_1(pol_theta, x) # 1 x n
    dpi_dtheta = POL_GRAD(policy_x, pol_theta, x) # n x p
    return dc_dpi.dot(dpi_dtheta).dot(W)




'''
find value of centered estimator, evaluated against a benchmark policy which assigns
Pi(x) = 1 w.p p_1 for all x
'''
def centered_around_p1(a_bnd, b_bnd, Y_T, pi_1, p_1):
    return find_opt_robust_ipw_val(np.multiply(Y_T, (pi_1 - p_1)), a_bnd, b_bnd, shorter=True)

def plot_W_GDS(p_ths, W_GDs):
    plot(p_ths, W_GDs[:,0])
    for i in range(len(p_ths)):
        plot([p_ths[i]-0.5, p_ths[i]+0.5], [W_GDs[i,0]-W_GDs[i,1]*0.5, W_GDs[i,0]+W_GDs[i,1]*0.5], c='b',alpha=0.1)

''' test gradient fn for th, given vector of assignments p_1
'''
def test_subgrad_for_th(p_th, p_1, PI_1, POL_GRAD, x, y, t01):
    n = x.shape[0]; pi_1 = PI_1(np.asarray([p_th]), x).flatten(); t=get_sgn_0_1(t01);
    [lda_opt, wghts, wghts_sum] = find_opt_weights_shorter(np.multiply(y*t, pi_1 - p_1), a_bnd, b_bnd)
    grad = get_implicit_grad_centered(p_th, PI_1, POL_GRAD, x, y, t01, wghts/wghts.sum())
    return [lda_opt,grad]

''' test gradient fn for th, regret against the anti-policy -Pi
'''
def test_subgrad_for_anti(p_th, p_1, PI_1, POL_GRAD, x, y, t01):
    n = x.shape[0]; pi_1 = PI_1(np.asarray([p_th]), x).flatten(); t=get_sgn_0_1(t01);
    [lda_opt, wghts, wghts_sum] = find_opt_weights_shorter(np.multiply(y*t, pi_1 - p_1), a_bnd, b_bnd)
    grad = get_implicit_grad_centered_anti_pi(p_th, PI_1, POL_GRAD, x, y, t01, wghts/wghts.sum())
    return [lda_opt,grad]

########### Data generation tools ########################################################################
"""
"""


    # generate propensity model
def real_prop(x, beta_prop):
    n = x.shape[0]; d = x.shape[1]
    prop = np.zeros(n)
    for i in range(n):
        prop[i] = np.exp(np.dot(beta_prop[0:d], x[i,:]) + beta_prop[-1] )/ (1 + np.exp(np.dot(beta_prop[0:d], x[i,:]) + beta_prop[-1] ))
    return prop
'''
requires specifying globals
HIGH_FREQ_N = 5
FREQ = 20
'''
def generate_data_nd(mu_x, sigma_x_mat, n, beta_cons, beta_x, beta_x_T, TRUE_PROP_BETA, beta_x_high_freq):
#     x = np.random.normal(mu_x, sigma_x, size = n)
    # generate n datapoints from the same multivariate normal distribution
    x = np.random.multivariate_normal(mean = mu_x, cov= sigma_x_mat, size = n )
    true_Q = real_prop(x, TRUE_PROP_BETA)
    T = np.array(np.random.uniform(size=n) < true_Q).astype(int).flatten()
    T = T.reshape([n,1])

    clf = LogisticRegression(); clf.fit(x, T)
    propensities = clf.predict_proba(x)
    print clf.coef_
    T_sgned = np.asarray([ 1 if T[i] == 1 else -1 for i in range(n)]).flatten()

    y_sigma = 0.5
    nominal_propensities_pos = propensities[:,1]; nominal_propensities_null = propensities[:,0]
    Q = np.asarray( [nominal_propensities_pos[i] if T[i] == 1 else nominal_propensities_null[i] for i in range(n)] )
    q = Q
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = T[i]*beta_cons + np.dot(beta_x.T, x[i,:]) + np.dot(beta_x_T.T, x[i,:]*T[i]) + np.dot(beta_x_high_freq.T, np.sin(x[i,0:HIGH_FREQ_N]*FREQ)*T[i])

    white_noise_coef = 0.2
    # add random noise
    Y += np.random.multivariate_normal(mean = np.zeros(n), cov=white_noise_coef * np.eye(n))
    Y_OFFSET = np.abs(np.min(Y))
    Y = Y + Y_OFFSET
    T = T.flatten()
#     px = x.reshape([n,p]); x_augmented = np.hstack([x, np.ones([n,1])])
    T_colors = [ 'r' if T[i] == 0 else 'b' for i in range(n)]
#     x_poly = np.hstack([x, x**2, x**3, np.ones([n,1])])
#     x_u_poly = np.hstack([x_poly[:,0:3], u.reshape([n,1])])
    return [x, T, Y, true_Q, clf, T_colors, Y_OFFSET]

########################################################################################

### Scale continuous
def scale_continuous(train_dict,test_dict):
    continuous = np.asarray([len(np.unique(train_dict['X'][:,j])) for j in range(train_dict['X'].shape[1])]) > 10
    print np.where(continuous)[0]
    def scale_columns(x_train_col, x_test_col):
        mn = np.mean(x_train_col); sd = np.std(x_train_col)
        x_train_col = (x_train_col*1.0 - mn*1.0)/sd
        x_test_col = (x_test_col*1.0 - mn)*1.0/sd
        return [x_train_col, x_test_col]

    for ind in np.where(continuous)[0]:
        [x_train_col, x_test_col] = scale_columns(train_dict['X'][:,ind], test_dict['X'][:,ind])
        train_dict['X'][:,ind] = x_train_col
        test_dict['X'][:,ind] = x_test_col
        return [train_dict, test_dict]


''' add new versions of items to data dict '''
def subsample_traindict(train_dict,TEST_FRAC):
    train_ind, test_ind = train_test_split(range(len(train_dict['Y'])), test_size=TEST_FRAC)
    dictlabels = ['T', 'Y', 'Yhf', 'prop_T' ]
    train_dict['X'] = train_dict['X'][train_ind,:]
    for ind,key in enumerate(dictlabels):
        train_dict[key] = train_dict[key][train_ind];
    # x_, x_test, t_sgned_, t_sgned_test, y_, y_test, yhf_, yhftest_, nominal_Q_, nominal_Q_test, train_ind, test_ind \
    # = train_test_split(train_dict['X'], train_dict['T'], train_dict['Y'],train_dict['Yhf'], train_dict['prop_T'], range(len(train_dict['Y'])), test_size=TEST_FRAC)
    # new_list = [x_, t_sgned_, y_,yhf_, nominal_Q_, train_ind]
    # dictlabels = [ 'X', 'T', 'Y', 'Yhf', 'q0' ] # this ends up permuting the estimated propensities
    # for ind,key in enumerate(dictlabels):
    #     train_dict[key] = new_list[ind];
    return [train_dict, train_ind]

def scale_columns(x_train_col, x_test_col):
    mn = np.mean(x_train_col); sd = np.std(x_train_col)
    x_train_col = (x_train_col*1.0 - mn*1.0)/sd
    x_test_col = (x_test_col*1.0 - mn)*1.0/sd
    return [x_train_col, x_test_col]

def scale_dicts(train_dict, test_dict):
    continuous = np.asarray([len(np.unique(train_dict['X'][:,j])) for j in range(train_dict['X'].shape[1])]) > 10
    for ind in np.where(continuous)[0]:
        [x_train_col, x_test_col] = scale_columns(train_dict['X'][:,ind], test_dict['X'][:,ind])
        train_dict['X'][:,ind] = x_train_col
        test_dict['X'][:,ind] = x_test_col
    return [train_dict, test_dict]
