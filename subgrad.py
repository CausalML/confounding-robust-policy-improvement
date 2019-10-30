"""
@author: Angela Zhou
"""
import numpy as np
import gurobipy as gp
from unconfoundedness_fns import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
import random
import sys



def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

''' get gradient of pi wrt theta for logistic policy
where pi = Pr[pi=1|x]
'''
def qk_dpi_dtheta(pi_1, pol_theta, x):
    n = len(pi_1); dg_dz = np.multiply(pi_1,np.ones(len(pi_1))-pi_1)
    return np.multiply(dg_dz[:,np.newaxis], x)
    #return np.diag(dg_dz).dot( x ).reshape(x.shape) # n x p: enforce jic


# """ get gradient of pi wrt theta for logistic policy
# where pi = Pr[pi=1|x]
# """
# def qk_dpi_dtheta_mt(pi_1, pol_theta, x):
#     n = len(pi_1);
#     K = pol_theta.shape[1]
#     grad = np.zeros(pol_theta.shape)
#     for a in range(K):
#         dg_dz = np.multiply(pi_1[:,a],np.ones(n)-pi_1[:,a])
#         grad[:,a] = np.dot(dg_dz[:, np.newaxis].T, x)
#     return grad

""" get gradient of pi wrt theta for logistic policy
where pi = Pr[pi=1|x]
"""
def qk_dpi_dtheta_mt_scalar(pi_1, x):
    return pi_1*(1-pi_1) * x

"""
read in callbacks for derivative of pi given theta and optimal w, t
PI_1, POL_GRAD (returns (p x 1) vector)
take in ** normalized weights W **
"""
def get_implicit_grad_centered(pol_theta, PI_1, POL_GRAD, x, Y, t01, W):
    # if need to get active index set
    # rescaled weights in original
    n = len(W); T_sgned = get_sgn_0_1(t01)
    constants = np.multiply(Y, np.multiply(T_sgned,W))
    policy_x = PI_1(pol_theta, x) # 1 x n
    dpi_dtheta = POL_GRAD(policy_x, pol_theta, x) # n x p
    if x.ndim > 1:
        return np.multiply(constants[:,np.newaxis], dpi_dtheta).sum(axis=0)
    else:
        return np.sum(np.multiply(constants,dpi_dtheta))

"""
returns gradient for multiple treatments case
read in callbacks for derivative of pi given theta and optimal w, t
theta is a d x K array 
PI_1 returns probability pi = Pr[ A_i | x ] under policy   
POL_GRAD returns gradient for observed A_i (returns (p x 1) vector)
take in *** normalized weights W ***
"""
def get_implicit_grad_centered_mt(pol_theta, PI_1, POL_GRAD_scalar, x, Y, t01, W):
    n = len(W); #t_levels=np.unique(t01)
    constants = np.multiply(Y, W)
    policy_x = PI_1(pol_theta, x, t01)
        # np.asarray([ PI_1(pol_theta[:,k], x) for k in t01 ]).flatten() # k x n
    dpi_dtheta = np.asarray([ POL_GRAD_scalar( policy_x[i], x[i,:] ) for i in range(n) ] )
    # call qk_dpi_dtheta for pol_theta[a_i] for each datapoint
    K = pol_theta.shape[1]
    grad = np.zeros(pol_theta.shape)
    for a in range(K):
        if x.ndim > 1:
            grad[:,a] = np.multiply(constants[(t01 == a), np.newaxis], dpi_dtheta[t01 == a,:]).sum(axis=0)
        else:
            grad[:,a] = np.sum(np.multiply(constants[t01 == a], dpi_dtheta[t01 == a]))
    # print grad.shape
    return grad

"""
find value of centered estimator, evaluated against a benchmark policy which assigns
Pi(x) = 1 w.p p_1 for all x
"""
def centered_around_p1(a_bnd, b_bnd, Y_T, pi_1, p_1):
    return find_opt_robust_ipw_val(np.multiply(Y_T, (pi_1 - p_1)), a_bnd, b_bnd, shorter=True)


def plot_W_GDS(p_ths, W_GDs):
    plot(p_ths, W_GDs[:,0])
    for i in range(len(p_ths)):
        plot([p_ths[i]-0.5, p_ths[i]+0.5], [W_GDs[i,0]-W_GDs[i,1]*0.5, W_GDs[i,0]+W_GDs[i,1]*0.5], c='b',alpha=0.1)

""" test gradient fn for th, given vector of assignments p_1
"""
def test_subgrad_for_th(p_th, p_1, PI_1, POL_GRAD, x, y, t01):
    n = x.shape[0]; pi_1 = PI_1(np.asarray([p_th]), x).flatten(); t=get_sgn_0_1(t01);
    [lda_opt, wghts, wghts_sum] = find_opt_weights_shorter(np.multiply(y*t, pi_1 - p_1), a_bnd, b_bnd)
    grad = get_implicit_grad_centered(p_th, PI_1, POL_GRAD, x, y, t01, wghts/wghts.sum())
    return [lda_opt,grad]

""" test gradient fn for th, regret against the anti-policy -Pi
"""
def test_subgrad_for_anti(p_th, p_1, PI_1, POL_GRAD, x, y, t01):
    n = x.shape[0]; pi_1 = PI_1(np.asarray([p_th]), x).flatten(); t=get_sgn_0_1(t01);
    [lda_opt, wghts, wghts_sum] = find_opt_weights_shorter(np.multiply(y*t, pi_1 - p_1), a_bnd, b_bnd)
    grad = get_implicit_grad_centered_anti_pi(p_th, PI_1, POL_GRAD, x, y, t01, wghts/wghts.sum())
    return [lda_opt,grad]



""" centered problem with weights
assume given pi_1 vector
max (pi_1 - p_1)YT W
"""
def get_general_interval_wghts_algo_centered_TV_prob(gamma, Y, a_, b_, fq, quiet=True):
    wm = 1/fq; wm_sum=wm.sum(); n = len(Y)
    wm = wm/wm_sum # normalize propensities
    # assume estimated propensities are probs of observing T_i
    y = Y; weights = np.zeros(n);
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w = [m.addVar(obj = -yy, lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    d = [m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    m.update()
    m.addConstr(gp.quicksum(w)==1)
    m.addConstr(gp.quicksum(d)<=gamma*t)
    for i in range(len(y)):
        m.addConstr(w[i] <= b_[i] * t/wm_sum)
        m.addConstr(w[i] >= a_[i] * t/wm_sum)
        m.addConstr(d[i] >=   w[i] - t*wm[i])
        m.addConstr(d[i] >= - w[i] + t*wm[i])
    m.optimize()
    wghts = np.asarray([ ww.X for ww in w ]) # would like to have failsafe for not being able to optimize
    return [-m.ObjVal,wghts,t.X/wm_sum]

"""
read in callbacks for derivative of pi given theta and optimal w, t
Y(Pi) - Y(-Pi)
PI_1, POL_GRAD (returns (p x 1) vector)
take in ** normalized weights W **
####
DEPRECATED beceause slow/memory using 
Use get_implicit_grad_centered instead
"""
# def get_implicit_grad_centered_anti_pi(pol_theta, PI_1, POL_GRAD, x, Y, t01, W):
#     n = len(W); T_sgned = get_sgn_0_1(t01)
#     dc_dpi = np.diag(2*Y*T_sgned)
#     policy_x = PI_1(pol_theta, x) # 1 x n
#     dpi_dtheta = POL_GRAD(policy_x, pol_theta, x) # n x p
#     return dc_dpi.dot(dpi_dtheta).T.dot(W) #! double check

def logistic_pol_asgn(theta, x):
    ''' Requires an intercept term
    '''
    n = x.shape[0]
    theta = theta.flatten()
    if len(theta) == 1:
        logit = np.multiply(x, theta).flatten()
    else:
        logit = np.dot(x, theta).flatten()
    LOGIT_TERM_POS = np.ones(n)*1.0 / ( np.ones(n) + np.exp( -logit ))
    return LOGIT_TERM_POS
''' wrapper to align arguments with the budgeted case
'''
def opt_wrapper(gamma, y, a_bnd, b_bnd, fq):
    return find_opt_weights_shorter(y, a_bnd, b_bnd)

def anti_p_1(pi_1):
    return np.ones(len(pi_1))-pi_1 # anti policy
def ctrl_p_1(pi_1):
    return np.zeros(len(pi_1))
def tmnt_p_1(pi_1):
    return np.ones(len(pi_1))

'''
input: arbitrary policy vector (to get shape) 
return 1 where treatment pattern is 0 (control
'''
def ctrl_p_1_mt(x,t01):
    # treatment probs 0
    pi_0 = np.zeros(x.shape[0])
    pi_0[t01==0] = 1 #  control treatment with probablity 1
    return pi_0




""" subgrad descent template algo
automatically augments data !
take in theta_0, # rounds
WGHTS_: fn obtaining optimal weights
GRAD_: fn to obtain parametric subgradient
POL_GRAD: gradient of parametrized policy wrt parameters
PI_1: return prob of pi(x) = 1
p_1: pi_0 probability of t = 1
"""
def grad_descent(th, N_RNDS, WGHTS_, GRAD_, POL_GRAD, PI_1, P_1, x, t01, fq, y,
 a_, b_, gamma,eta_0,logging=False,step_schedule=0.5,linesearch=True):
    n = x.shape[0]; t = get_sgn_0_1(t01)
    assert all(len(arr) == n for arr in [x,t01,fq,y,a_,b_])
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    # x_aug = np.hstack([x, np.ones([n,1])]);
    risks = np.zeros(N_RNDS); THTS = [None]*N_RNDS; PSTARS = np.zeros([N_RNDS, n]); losses = [None]*N_RNDS;

    def opt_wrapper_here(th):#linesearch wrapper
        pi_1 = PI_1(th, x_aug).flatten();
        p_1 = P_1(pi_1);
        [lda_opt, wghts, wghts_sum] = WGHTS_(gamma, np.multiply(y*t, pi_1 - p_1), a_, b_, fq)
        return lda_opt

    for k in range(N_RNDS) :
        # print k; sys.stdout.flush()
        eta_t = eta_0 * 1.0/np.power((k+1)*1.0, step_schedule); eta_t_og = eta_t
        pi_1 = PI_1(th, x_aug).flatten();
        p_1 = P_1(pi_1);
        [lda_opt, wghts, wghts_sum] = WGHTS_(gamma, np.multiply(y*t, pi_1 - p_1), a_, b_, fq)
        subgrad = GRAD_(th, PI_1, POL_GRAD, x_aug, y, t, wghts*1.0 / wghts_sum )
        if (linesearch==True): # Armijo LineSearch
            # print 'ls'; sys.stdout.flush()
            ALS = ArmijoLineSearch(tfactor = 0.2)
            d = -subgrad
            slope = np.dot(subgrad, d)
            eta_t = ALS.search(opt_wrapper_here, th, d, slope, f = lda_opt)
        # print eta_t, 'eta_t'; sys.stdout.flush()
        if eta_t is not None:
            th = th - eta_t * subgrad
        else:
            th = th - eta_t_og * subgrad
#         oosrisks[k] = np.mean( PI_1(th, x_test_aug)*y_test[t01==1]/true_Q_test[t01==1]+(1-PI_1(th, x_test_aug))*y_test[t01==0]/true_Q_test[t01==0] );
        THTS[k] = th; PSTARS[k,:] = wghts.flatten(); losses[k] = lda_opt
    return [losses, THTS, PSTARS]

""" subgrad descent template algo
This is the sharp version
which computes weights separately in each T=1, T=-1 component
automatically augments data !
take in theta_0, # rounds
WGHTS_: fn obtaining optimal weights
This weight function is overloaded to accommodate weight functions that take in extra parameters
about the uncertainty set
GRAD_: fn to obtain parametric subgradient
POL_GRAD: gradient of parametrized policy wrt parameters
PI_1: return prob of pi(x) = 1
p_1: pi_0 probability of t = 1
"""
def grad_descent_sharp(th, N_RNDS, WGHTS_, GRAD_, POL_GRAD, POL_PROB_1, BASELINE_POL, x, t01, fq, y,
 a_, b_, gamma,eta_0,logging=False,step_schedule=0.5,linesearch=True):
    n = x.shape[0]; t = get_sgn_0_1(t01)
    t_levels = np.unique(t01)
    assert all(len(arr) == n for arr in [x,t01,fq,y,a_,b_])
    # Check if x data contains an intercept: only retain for backwards compatibility:
    # If last column is all ones, don't augment
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    risks = np.zeros(N_RNDS); THTS = [None]*N_RNDS; PSTARS = np.zeros([N_RNDS, n]); losses = [None]*N_RNDS;
    if not hasattr(gamma, "__len__"):
        gammas = gamma * np.ones(len(t_levels))
    else:
        gammas = gamma
    def opt_wrapper_here(th):#linesearch wrapper
        pi_1 = POL_PROB_1(th, x_aug).flatten();
        p_1 = BASELINE_POL(pi_1);
        [lda_opt_neg1, wghts_neg1, wghts_sum_neg1] = WGHTS_(gammas[0], np.multiply(-1*y[t == -1], pi_1[t == -1] - p_1[t == -1]), a_[t == -1], b_[t == -1], fq[t == -1])
        [lda_opt_1, wghts_1, wghts_sum_1] = WGHTS_(gammas[1], np.multiply(y[t == 1], pi_1[t == 1] - p_1[t == 1]), a_[t == 1], b_[t == 1], fq[t == 1])

        wghts_total[t==1] = wghts_1*1.0 / wghts_sum_1; wghts_total[t==-1] = wghts_neg1*1.0 / wghts_sum_neg1;

        return lda_opt_1 + lda_opt_neg1

    for k in range(N_RNDS) :
        eta_t = eta_0 * 1.0/np.power((k+1)*1.0, step_schedule); eta_t_og = eta_t
        pi_1 = POL_PROB_1(th, x_aug).flatten();
        p_1 = BASELINE_POL(pi_1);
        # Modification for sharpness: Call the weight subroutine within treated and untreated groups separately
        wghts_total = np.zeros(len(t)); subgrad_total = np.zeros(len(t))
        # compute for T=1
        #TODO: modify for general case, mult T
        [lda_opt_1, wghts_1, wghts_sum_1] = WGHTS_(gammas[1], np.multiply(y[t == 1], pi_1[t == 1] - p_1[t == 1]), a_[t == 1], b_[t == 1], fq[t == 1])
        # compute for T=-1
        [lda_opt_neg1, wghts_neg1, wghts_sum_neg1] = WGHTS_(gammas[0], np.multiply(-1*y[t == -1], pi_1[t == -1] - p_1[t == -1]), a_[t == -1], b_[t == -1], fq[t == -1])
        wghts_total[t==1] = wghts_1*1.0 / wghts_sum_1; wghts_total[t==-1] = wghts_neg1*1.0 / wghts_sum_neg1;
        subgrad = GRAD_(th, POL_PROB_1, POL_GRAD, x_aug, y, t, wghts_total*1.0 / np.sum(wghts_total) )
        lda_opt = lda_opt_1 + lda_opt_neg1
        if (linesearch==True): # Armijo LineSearch
            ALS = ArmijoLineSearch(tfactor = 0.2, default = eta_t)
            d = -subgrad
            slope = np.dot(subgrad, d)
            eta_t = ALS.search(opt_wrapper_here, th, d, slope, f = lda_opt)
        # print eta_t, 'eta_t'; sys.stdout.flush()
        if eta_t is not None:
            th = th - eta_t * subgrad
        else:
            th = th - eta_t_og * subgrad
#         oosrisks[k] = np.mean( POL_PROB_1(th, x_test_aug)*y_test[t01==1]/true_Q_test[t01==1]+(1-PI_1(th, x_test_aug))*y_test[t01==0]/true_Q_test[t01==0] );
        THTS[k] = th; PSTARS[k,:] = wghts_total.flatten(); losses[k] = lda_opt
    return [losses, THTS, PSTARS]



# random restarts
''' refactored signature
anti_p_1 -> pi_0
'''
def opt_w_restarts_rf(N_RST, th, N_RNDS, WGHTS_, GRAD_, POL_GRAD, PI_1, pi_0,
X, T, Y, q0, a_bnd, b_bnd, gamma,
eta_0,logging=False,step_schedule=0.5, averaging = False, give_initial=False):
    ls = np.zeros(N_RST); ths = [None] *N_RST
    iterator = log_progress(range(N_RST),every=1) if logging else range(N_RST)
    for j in iterator:
        random.seed(j)
        if not give_initial:
            th_0 = np.random.randn(x.shape[1]+1);
        else:
            th_0 = th
        [oosrisks, losses, THTS, PSTARS] = grad_descent(th_0, N_RNDS, WGHTS_,
        GRAD_, POL_GRAD, PI_1, pi_0, X, T, Y, q0, a_bnd, b_bnd, gamma,
        eta_0,step_schedule)
        if averaging: #average losses: OGD rule
            ls[j] = np.mean(losses); ths[j] = sum(THTS)/len(THTS)
        else:
        # return the best so far, not last
            best_so_far = np.argmin(losses)
            ls[j]=losses[best_so_far]; ths[j] = THTS[best_so_far]
        if logging:
            plt.plot(range(N_RNDS), losses)
            plt.pause(0.05)
    return [ths[np.argmin(ls)], min(ls)] #return tht achieving min loss

# random restarts
def opt_w_restarts(N_RST, th,
                   N_RNDS, WGHTS_, GRAD_, POL_GRAD, POL_PROB_1, BASELINE_POL, # specify policy functions, gradients, weight function s
                   x, t, fq, y, a_bnd, b_bnd, gamma, eta_0, # give data
        logging=False, step_schedule=0.5, averaging = False, give_initial=False, sharp = False, **kwargs): # other opt settings
    ls = np.zeros(N_RST); ths = [None] *N_RST
    iterator = log_progress(range(N_RST),every=1) if logging else range(N_RST)
    for j in iterator:
        random.seed(j)
        if (give_initial) and (j==0):
            th_0 = th
        else:
            th_0 = np.random.randn(x.shape[1])*0.25;
        if sharp:
            # print 'using sharp estimator'
            [losses, THTS, PSTARS] = grad_descent_sharp(th_0, N_RNDS, WGHTS_, GRAD_, POL_GRAD, POL_PROB_1, BASELINE_POL, x, t, fq, y, a_bnd, b_bnd, gamma,eta_0,step_schedule)
        else:
            [losses, THTS, PSTARS] = grad_descent(th_0, N_RNDS, WGHTS_, GRAD_, POL_GRAD, POL_PROB_1, BASELINE_POL, x, t, fq, y, a_bnd, b_bnd, gamma,eta_0,step_schedule)
        if averaging: #average losses: OGD rule
            ls[j] = np.mean(losses); ths[j] = sum(THTS)/len(THTS)
        else:
            best_so_far = np.argmin(losses) # return the best so far, not last
            ls[j]=losses[best_so_far]; ths[j] = THTS[best_so_far]
        if logging:
            plt.plot(range(N_RNDS), losses)
            plt.pause(0.05)
    if logging:
        print ls, 'opt losses'
    return [ths[np.argmin(ls)], min(ls)] #return tht achieving min loss


""" Multiple treatments, optimize with restarts  
"""
def opt_w_restarts_mt(N_RST, th,
                   N_RNDS, WGHTS_, GRAD_, POL_GRAD, POL_PROB_1, BASELINE_POL, # specify policy functions, gradients, weight function s
                   x, t, fq, y, a_bnd, b_bnd, gamma, eta_0, # give data
        logging=False, step_schedule=0.5, averaging = False, give_initial=False, sharp = False, **kwargs): # other opt settings
    ls = np.zeros(N_RST); ths = [None] *N_RST
    n_ts = len(np.unique(t))
    iterator = log_progress(range(N_RST),every=1) if logging else range(N_RST)
    for j in iterator:
        random.seed(j)

        if (give_initial) and (j==0):
            th_0 = th
        else:
            th_0 = np.random.randn(x.shape[1],n_ts)*0.25;

        [losses, THTS, PSTARS] = grad_descent_sharp_mt(th_0, N_RNDS, WGHTS_, GRAD_, POL_GRAD, POL_PROB_1, BASELINE_POL, x, t, fq, y, a_bnd, b_bnd, gamma,eta_0,step_schedule)

        if averaging: #average losses: OGD rule
            ls[j] = np.mean(losses); ths[j] = sum(THTS)/len(THTS)
        else:
            best_so_far = np.argmin(losses) # return the best so far, not last
            ls[j]=losses[best_so_far]; ths[j] = THTS[best_so_far]
        if logging:
            plt.plot(range(N_RNDS), losses)
            plt.pause(0.05)
    if logging:
        print ls, 'opt losses'
    return [ths[np.argmin(ls)], min(ls)] #return tht achieving min loss

""" logistic pol assign observed treatment A_i
"""


""" 
subgrad descent template algo
This is the sharp version for multiple treatments 
which computes weights separately in each T=1, T=-1 component
automatically augments data !
take in theta_0, # rounds
WGHTS_: fn obtaining optimal weights
This weight function is overloaded to accommodate weight functions that take in extra parameters
about the uncertainty set
GRAD_: fn to obtain parametric subgradient
POL_GRAD: gradient of parametrized policy wrt parameters
POL_PROB_A: return prob of pi(x) = A_i (for observed A_i)
BASELINE_POL: pi_0 probability of pi_0 = A_i (for observed A_i)
"""
def grad_descent_sharp_mt(th, N_RNDS, WGHTS_, GRAD_, POL_GRAD, POL_PROB_A, BASELINE_POL, x, t01, fq, y,
 a_, b_, gamma,eta_0,logging=False,step_schedule=0.5,linesearch=False):
    n = x.shape[0]; #t = get_sgn_0_1(t01)
    t_levels = np.unique(t01)
    assert all(len(arr) == n for arr in [x,t01,fq,y,a_,b_])
    # Check if x data contains an intercept: only retain for backwards compatibility:
    # If last column is all ones, don't augment
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    risks = np.zeros(N_RNDS); THTS = [None]*N_RNDS; PSTARS = np.zeros([N_RNDS, n]); losses = [None]*N_RNDS;
    if not hasattr(gamma, "__len__"): # Default to repeating gamma if not specified
        gammas = gamma * np.ones(len(t_levels))
    else:
        gammas = gamma
    def opt_wrapper_here(th):#linesearch wrapper
        pi_1 = POL_PROB_A(th, x_aug, t01).flatten();
        p_1 = BASELINE_POL(pi_1, t01);
        wghts_total = np.zeros(len(t01)); lda_opt_total = 0
        for ind,t_l in range(t_levels):
            [lda_opt, wghts, wghts_sum] = WGHTS_(gammas[ind], np.multiply(y[t01 == t_l], pi_1[t01 == t_l] - p_1[t01 == t_l]), a_[t01 == t_l], b_[t01 == t_l], fq[t01 == t_l])
            wghts_total[t01==t_l] = wghts*1.0 / wghts_sum;
            lda_opt_total += lda_opt
        return lda_opt_total

    for k in range(N_RNDS) :
        eta_t = eta_0 * 1.0/np.power((k+1)*1.0, step_schedule); eta_t_og = eta_t
        pi_1 = POL_PROB_A(th, x_aug, t01).flatten();
        p_1 = BASELINE_POL(pi_1, t01);
        # Modification for sharpness:
        # Call the weight subroutine within each treatment partition separately
        wghts_total_norm = np.zeros(len(t01));
        lda_opt_total = 0
        #TODO: modify for general case, mult T
        for ind,t_l in enumerate(t_levels):
            # print 'pi_1', pi_1
            # print pi_1[t01 == t_l]
            # print 'th', th
            # print p_1[t01 == t_l]
            [lda_opt, wghts, wghts_sum] = WGHTS_(gammas[ind], np.multiply(y[t01 == t_l], pi_1[t01 == t_l] - p_1[t01 == t_l]), a_[t01 == t_l], b_[t01 == t_l], fq[t01 == t_l])
            wghts_total_norm[t01==t_l] = wghts*1.0 / wghts_sum;
            lda_opt_total += lda_opt

        subgrad = GRAD_(th, POL_PROB_A, POL_GRAD, x_aug, y, t01, wghts_total_norm )

        if (linesearch==True): # Armijo LineSearch
            ALS = ArmijoLineSearch(tfactor = 0.2, default = eta_t)
            d = -subgrad
            slope = np.dot(subgrad, d)
            eta_t = ALS.search(opt_wrapper_here, th, d, slope, f = lda_opt_total)
        if eta_t is not None:
            th = th - eta_t * subgrad
        else:
            th = th - eta_t_og * subgrad
        THTS[k] = th; PSTARS[k,:] = wghts_total_norm.flatten(); losses[k] = lda_opt_total
    return [losses, THTS, PSTARS]

''' objective evaluation for multiple treatments 
'''
def mt_obj_eval(th, *args):  # linesearch wrapper
    [WGHTS_, GRAD_, POL_GRAD, POL_PROB_A, BASELINE_POL, x, t01, fq, y, a_, b_, gamma] = args
    t_levels = np.unique(t01)
    if not hasattr(gamma, "__len__"): # Default to repeating gamma if not specified
        gammas = gamma * np.ones(len(t_levels))
    else:
        gammas = gamma
    n = x.shape[0]; #t = get_sgn_0_1(t01)
    assert all(len(arr) == n for arr in [x,t01,fq,y,a_,b_])
    # Check if x data contains an intercept: only retain for backwards compatibility:
    # If last column is all ones, don't augment
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    pi_1 = POL_PROB_A(th, x_aug, t01).flatten();
    p_1 = BASELINE_POL(pi_1, t01);
    wghts_total = np.zeros(len(t01));
    lda_opt_total = 0
    for ind, t_l in enumerate(t_levels):
        [lda_opt, wghts, wghts_sum] = WGHTS_(gammas[ind],
        np.multiply(y[t01 == t_l], pi_1[t01 == t_l] - p_1[t01 == t_l]),
        a_[t01 == t_l], b_[t01 == t_l], fq[t01 == t_l])
        wghts_total[t01 == t_l] = wghts * 1.0 / wghts_sum;
        lda_opt_total += lda_opt
    return lda_opt_total

''' gradient evaluation for multiple treatments 
'''
def mt_grad_eval(th, *args):  # linesearch wrapper
    [WGHTS_, GRAD_, POL_GRAD, POL_PROB_A, BASELINE_POL, x, t01, fq, y, a_, b_, gamma] = args
    n = x.shape[0]; #t = get_sgn_0_1(t01)
    t_levels = np.unique(t01)
    if not hasattr(gamma, "__len__"): # Default to repeating gamma if not specified
        gammas = gamma * np.ones(len(t_levels))
    else:
        gammas = gamma
    assert all(len(arr) == n for arr in [x,t01,fq,y,a_,b_])
    # Check if x data contains an intercept: only retain for backwards compatibility:
    # If last column is all ones, don't augment
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    wghts_total_norm = np.zeros(len(t01));
    lda_opt_total = 0
    # TODO: modify for general case, mult T
    pi_1 = POL_PROB_A(th, x_aug, t01).flatten();
    p_1 = BASELINE_POL(pi_1, t01);
    for ind, t_l in enumerate(t_levels):
        [lda_opt, wghts, wghts_sum] = WGHTS_(gammas[ind],
                                             np.multiply(y[t01 == t_l], pi_1[t01 == t_l] - p_1[t01 == t_l]),
                                             a_[t01 == t_l], b_[t01 == t_l], fq[t01 == t_l])
        wghts_total_norm[t01 == t_l] = wghts * 1.0 / wghts_sum;
        lda_opt_total += lda_opt
    subgrad = GRAD_(th, POL_PROB_A, POL_GRAD, x_aug, y, t01, wghts_total_norm)
    return subgrad

# def grad_descent_vanilla_ipw(th, N_RNDS, WGHTS_, GRAD_, POL_GRAD, PI_1, P_1, x, t, fq, y, a_, b_, gamma,eta_0,logging=False,step_schedule=0.5):
def grad_descent_vanilla_ipw(th, N_RNDS, POL_PROB_1, x, t, fq, y, eta_0,logging=False,step_schedule=0.5):
    n = x.shape[0];
    assert all(len(arr) == n for arr in [x,t,fq,y])
    # If last column is all ones, don't augment
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    risks = np.zeros(N_RNDS); THTS = [None]*N_RNDS; PSTARS = np.zeros([N_RNDS, n]); losses = [None]*N_RNDS; oosrisks = np.zeros(N_RNDS)
    fq_norm = fq/np.sum(fq)
    for k in range(N_RNDS) :
        eta_t = eta_0 * 1.0/np.power((k+1)*1.0, 0.3);
        pi_1 = POL_PROB_1(th, x_aug).flatten();
        pi_t = np.asarray( [pi_1[i] if t[i] == 1 else 1- pi_1[i] for i in range(n)] )
        loss = np.sum( y*pi_t/fq_norm  )
        pi_1_grad = qk_dpi_dtheta(pi_1, th, x_aug)
        pi_t_grad = np.asarray( [pi_1_grad[i] if t[i] == 1 else -pi_1_grad[i] for i in range(n)] )
        subgrad = (y/fq_norm).dot(pi_t_grad).T
        th = th - eta_t * subgrad
        THTS[k] = th; losses[k] = loss
    return [oosrisks, losses, THTS, PSTARS]

# random restarts
def opt_w_restarts_vanilla_ipw(N_RST, th, N_RNDS, POL_PROB_1, x, t, fq, y, eta_0,logging=False,step_schedule=0.5,**params):
    ls = np.zeros(N_RST); ths = [None] *N_RST
    iterator = log_progress(range(N_RST),every=1) if logging else range(N_RST)
    for j in iterator:
        # assume data has intercept
        th_0 = np.random.randn(x.shape[1]);
        [oosrisks, losses, THTS, PSTARS] = grad_descent_vanilla_ipw(th_0, N_RNDS, POL_PROB_1, x, t, fq, y, eta_0,logging=False,step_schedule=step_schedule)
        ls[j]=losses[-1]; ths[j] = THTS[-1]
        if logging:
            plt.plot(range(N_RNDS), losses)
            plt.pause(0.05)
    return [ths[np.argmin(ls)], min(ls)] #return tht achieving min loss

''' linear CATE projection 
'''
def grad_descent_linearproj_CATE(th, N_RNDS, POL_PROB_1, x, t, fq, y, eta_0,logging=False,step_schedule=0.5):
    n = x.shape[0];
    assert all(len(arr) == n for arr in [x,t,fq,y])
    # If last column is all ones, don't augment
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    risks = np.zeros(N_RNDS); THTS = [None]*N_RNDS; PSTARS = np.zeros([N_RNDS, n]); losses = [None]*N_RNDS; oosrisks = np.zeros(N_RNDS)
    fq_norm = fq/np.sum(fq)
    for k in range(N_RNDS) :
        eta_t = eta_0 * 1.0/np.power((k+1)*1.0, 0.3);
        pi_1 = POL_PROB_1(th, x_aug).flatten();
        # pi_t = np.asarray( [pi_1[i] if t[i] == 1 else 1- pi_1[i] for i in range(n)] )
        loss = np.sum( pi_1 * y ) # assume y is CATE estimate
        pi_1_grad = qk_dpi_dtheta(pi_1, th, x_aug)
        # pi_t_grad = np.asarray( [pi_1_grad[i] if t[i] == 1 else -pi_1_grad[i] for i in range(n)] )
        subgrad = (y).dot(pi_1_grad).T
        th = th - eta_t * subgrad
        THTS[k] = th; losses[k] = loss
    return [oosrisks, losses, THTS, PSTARS]


def opt_w_restarts_generic(GRAD_DESCENT, N_RST, N_RNDS, POL_PROB_1, x, t, fq, y, eta_0,logging=False,step_schedule=0.5,**params):
    ls = np.zeros(N_RST); ths = [None] *N_RST
    iterator = log_progress(range(N_RST),every=1) if logging else range(N_RST)
    for j in iterator:
        # assume data has intercept
        th_0 = np.random.randn(x.shape[1]);
        [oosrisks, losses, THTS, PSTARS] = GRAD_DESCENT(th_0, N_RNDS, POL_PROB_1, x, t, fq, y, eta_0,logging=False,step_schedule=step_schedule)
        ls[j]=losses[-1]; ths[j] = THTS[-1]
        if logging:
            plt.plot(range(N_RNDS), losses)
            plt.pause(0.05)
    return [ths[np.argmin(ls)], min(ls)] #return tht achieving min loss


# def grad_descent_vanilla_ipw(th, N_RNDS, WGHTS_, GRAD_, POL_GRAD, PI_1, P_1, x, t, fq, y, a_, b_, gamma,eta_0,logging=False,step_schedule=0.5):
def grad_descent_vanilla_ipw_centered(th, N_RNDS, PI_1, P_1, x, t, fq, y, eta_0,logging=False,step_schedule=0.5):
    n = x.shape[0];
    assert all(len(arr) == n for arr in [x,t,fq,y])
    x_aug = np.hstack([x, np.ones([n,1])]);
    risks = np.zeros(N_RNDS); THTS = [None]*N_RNDS; PSTARS = np.zeros([N_RNDS, n]); losses = [None]*N_RNDS; oosrisks = np.zeros(N_RNDS)
    fq_norm = fq #/np.sum(fq)
    for k in range(N_RNDS) :
        eta_t = eta_0 * 1.0/np.power((k+1)*1.0, 0.3);
        pi_1 = PI_1(th, x_aug).flatten();
        p_1 = P_1(pi_1);
        pi_t = np.asarray( [pi_1[i] if t[i] == 1 else 1- pi_1[i] for i in range(n)] )
        loss = np.sum( y*(pi_t-p_1)/fq_norm  )
        pi_1_grad = qk_dpi_dtheta(pi_1, th, x_aug)
        pi_t_grad = np.asarray( [pi_1_grad[i] if t[i] == 1 else -pi_1_grad[i] for i in range(n)] )
        subgrad = (y/fq_norm).dot(pi_t_grad).T
        th = th - eta_t * subgrad
        THTS[k] = th; losses[k] = loss
    return [oosrisks, losses, THTS, PSTARS]

def grad_descent_policynorm_ipw(th, N_RNDS, PI_1, P_1, x, t, fq, y, eta_0,logging=False,step_schedule=0.5):
    n = x.shape[0];
    assert all(len(arr) == n for arr in [x,t,fq,y])
    x_aug = np.hstack([x, np.ones([n,1])]);
    risks = np.zeros(N_RNDS); THTS = [None]*N_RNDS; PSTARS = np.zeros([N_RNDS, n]); losses = [None]*N_RNDS; oosrisks = np.zeros(N_RNDS)
    fq_norm = fq#/np.sum(fq)
    for k in range(N_RNDS) :
        eta_t = eta_0 * 1.0/np.power((k+1)*1.0, 0.3);
        pi_1 = PI_1(th, x_aug).flatten();
        pi_t = np.asarray( [pi_1[i] if t[i] == 1 else 1- pi_1[i] for i in range(n)] )
        policy_norm_denom = np.sum( pi_t / fq_norm )
        loss = np.sum( y*pi_t/fq_norm  ) / policy_norm_denom # self normalized version
        pi_t_grad = np.asarray( [pi_1_grad[i] if t[i] == 1 else -pi_1_grad[i] for i in range(n)] )
        subgrad = (y/fq_norm).dot(pi_t_grad).T / policy_norm_denom - (loss/policy_norm_denom)* (1/fq_norm).dot(pi_t_grad).T
        th = th - eta_t * subgrad
        THTS[k] = th; losses[k] = loss
    return [oosrisks, losses, THTS, PSTARS]
# random restarts
def opt_w_restarts_vanilla_ipw_centered(N_RST, th, N_RNDS, PI_1, P_1, x, t, fq, y, eta_0,
logging=False,step_schedule=0.5,normalized_policy=False):
    ls = np.zeros(N_RST); ths = [None] *N_RST
    iterator = log_progress(range(N_RST),every=1) if logging else range(N_RST)
    for j in iterator:
        th_0 = np.random.randn(x.shape[1]+1);
        if normalized_policy:
            [oosrisks, losses, THTS, PSTARS] = grad_descent_policynorm_ipw(th_0, N_RNDS, PI_1, P_1, x, t, fq, y, eta_0,logging=False,step_schedule=step_schedule)
        else:
            [oosrisks, losses, THTS, PSTARS] = grad_descent_vanilla_ipw(th_0, N_RNDS, PI_1, P_1, x, t, fq, y, eta_0,logging=False,step_schedule=step_schedule)
        ls[j]=losses[-1]; ths[j] = THTS[-1]
        if logging:
            plt.plot(range(N_RNDS), losses)
            plt.pause(0.05)
    return [ths[np.argmin(ls)], min(ls)] #return tht achieving min loss


# random restarts
def opt_w_restarts_sc_min(N_RST, FN_, JAC_, d, args_):
    ls = np.zeros(N_RST); ths = [None] * N_RST
    for j in range(N_RST):
        th_0 = np.random.randn(d+1);
        res = minimize(FN_, th_0, jac = JAC_, method='L-BFGS-B', args = tuple(args_),options={'disp': True})
        ls[j]=FN_(res.x, *args_); ths[j] = res.x
    return [ths[np.argmin(ls)], min(ls)] #return tht achieving min loss


""" general template algorithm
"""
def grad_descent_template(th, N_RNDS, LOSS_, GRAD_, POL_GRAD, PI_1, P_1, x, t01, fq, y, a_, b_, gamma,eta_0,logging=False,step_schedule=0.5):
    n = x.shape[0]; t = get_sgn_0_1(t01)
    assert all(len(arr) == n for arr in [x,t01,fq,y,a_,b_])
    x_aug = np.hstack([x, np.ones([n,1])]);
    risks = np.zeros(N_RNDS); THTS = [None]*N_RNDS; PSTARS = np.zeros([N_RNDS, n]); losses = [None]*N_RNDS; oosrisks = np.zeros(N_RNDS)
    for k in range(N_RNDS) :
        eta_t = eta_0 * 1.0/np.power((k+1)*1.0, 0.3);
        pi_1 = PI_1(th, x_aug).flatten();
        p_1 = P_1(pi_1);
        loss = LOSS_(gamma, np.multiply(y*t, pi_1 - p_1), a_, b_, fq)
        subgrad = GRAD_(th, PI_1, POL_GRAD, x_aug, y, t, wghts*1.0 / wghts_sum )
        th = th - eta_t * subgrad
#         oosrisks[k] = np.mean( PI_1(th, x_test_aug)*y_test[t01==1]/true_Q_test[t01==1]+(1-PI_1(th, x_test_aug))*y_test[t01==0]/true_Q_test[t01==0] );
        THTS[k] = th; PSTARS[k,:] = wghts.flatten(); losses[k] = lda_opt
        if k > 0:
            if np.isclose(lda_opt, losses[k-1], atol = 0.0001):
                return [oosrisks, losses[0:k], THTS[0:k], PSTARS]
    return [oosrisks, losses, THTS, PSTARS]


#### Functions suitable for use with lbfgs
def opt_wrapper_sc(th, *args):
    C = 0.05
    PI_1 = args[0]; x_aug=args[1]; y=args[2]; t_sgned=args[3]; fq = args[4]; P_1 = args[5]; a_bnd = args[6]; b_bnd = args[7]; n = len(y)
    pi_1 = PI_1(th, x_aug).flatten(); p_1 = P_1(pi_1);
    [lda_opt, wghts, wghts_sum]= find_opt_weights_shorter(np.multiply(y*t_sgned, pi_1 - p_1), a_bnd, b_bnd)
    return lda_opt + C*np.linalg.norm(th,2)**2

def get_implicit_grad_centered_sc(th, *args):
    C = 0.05
    PI_1 = args[0]; x_aug=args[1]; y=args[2]; t_sgned=args[3]; fq = args[4]; P_1 = args[5]; a_bnd = args[6]; b_bnd = args[7]; POL_GRAD = args[8]; n = len(y)
    pi_1 = PI_1(th, x_aug) # 1 x n
    p_1 = P_1(pi_1);
    [lda_opt, wghts, wghts_sum] = find_opt_weights_shorter(np.multiply(y*t_sgned, pi_1 - p_1), a_bnd, b_bnd)
    W = wghts / wghts_sum
    # more efficiently compute constants:
    constants = np.multiply(y, np.multiply(t_sgned,W))
    policy_x = PI_1(th, x_aug) # 1 x n
    dpi_dtheta = POL_GRAD(policy_x, th, x_aug) # n x p
    if x_aug.ndim > 1:
        return np.multiply(constants[:,np.newaxis], dpi_dtheta).sum(axis=0)
    else:
        return np.sum(np.multiply(constants,dpi_dtheta))

def get_implicit_grad_centered_anti_pi_sc(th, *args):
    C = 0.05
    PI_1 = args[0]; x_aug=args[1]; y=args[2]; t_sgned=args[3]; fq = args[4]; P_1 = args[5]; a_bnd = args[6]; b_bnd = args[7]; POL_GRAD = args[8]; n = len(y)
    pi_1 = PI_1(th, x_aug) # 1 x n
    p_1 = P_1(pi_1);
    [lda_opt, wghts, wghts_sum] = find_opt_weights_shorter(np.multiply(y*t_sgned, pi_1 - p_1), a_bnd, b_bnd)
    W = wghts / wghts_sum
    dc_dpi = np.diag(2*y*t_sgned)
    dpi_dtheta = POL_GRAD(pi_1, th, x_aug) # n x p
    return dc_dpi.dot(dpi_dtheta).T.dot(W) + 2*C*th #! double check

""" gamma infinity subgradient descent
"""
def grad_descent_gam_inf(th, N_RNDS, WGHTS_, GRAD_, POL_GRAD, PI_1, P_1, x, t01, fq, y, a_, b_, gamma,eta_0,logging=False,step_schedule=0.5):
    n = x.shape[0]; t = get_sgn_0_1(t01)
    assert all(len(arr) == n for arr in [x,t01,fq,y,a_,b_])
    x_aug = np.hstack([x, np.ones([n,1])]);
    risks = np.zeros(N_RNDS); THTS = [None]*N_RNDS; PSTARS = np.zeros([N_RNDS, n]); losses = [None]*N_RNDS; oosrisks = np.zeros(N_RNDS)
    for k in range(N_RNDS) :
        eta_t = eta_0 * 1.0/np.power((k+1)*1.0, step_schedule);
        pi_1 = PI_1(th, x_aug).flatten();
        p_1 = P_1(pi_1);
        options = np.multiply(y*t, pi_1 - p_1)
        lda_opt = np.max( options ); best_ind = np.argmax( options )
        # [lda_opt, wghts, wghts_sum] = WGHTS_(gamma, np.multiply(y*t, pi_1 - p_1), a_, b_, fq)
        subgrad = y[best_ind]*t[best_ind] * pi_1[best_ind] * (1 - pi_1[best_ind]) * x_aug[best_ind, :]
        th = th - eta_t * subgrad
#         oosrisks[k] = np.mean( PI_1(th, x_test_aug)*y_test[t01==1]/true_Q_test[t01==1]+(1-PI_1(th, x_test_aug))*y_test[t01==0]/true_Q_test[t01==0] );
        THTS[k] = th;
        # PSTARS[k,:] = wghts.flatten();
        losses[k] = lda_opt
    return [oosrisks, losses, THTS, PSTARS]

def opt_w_restarts_gam_inf(N_RST, th, N_RNDS, WGHTS_, GRAD_, POL_GRAD, PI_1, anti_p_1, x, t_sgned, fq, y, a_bnd, b_bnd, gamma,eta_0,logging=False,step_schedule=0.5):
    ls = np.zeros(N_RST); ths = [None] *N_RST
    iterator = log_progress(range(N_RST),every=1) if logging else range(N_RST)
    for j in iterator:
        th_0 = np.random.randn(x.shape[1]+1);
        [oosrisks, losses, THTS, PSTARS] = grad_descent_gam_inf(th_0, N_RNDS, WGHTS_, GRAD_, POL_GRAD, PI_1, anti_p_1, x, t_sgned, fq, y, a_bnd, b_bnd, gamma,eta_0,step_schedule)
        # return the best so far, not last
        best_so_far = np.argmin(losses)
        ls[j]=losses[best_so_far]; ths[j] = THTS[best_so_far]
        if logging:
            plt.plot(range(N_RNDS), losses)
            plt.pause(0.05)
    return [ths[np.argmin(ls)], min(ls)] #return tht achieving min loss

'''
Evaluate loss of policy th under different bounds a_, b_ (corresponding to a different value of Gamma)
PI_1 policy prob: return probability pi(x) = 1
p_1: baseline probability
a_, b_: bounds corresponding to Gamma being assessed (e.g. Un)
gamma: effective gamma bound (no-op for unbudgeted uncertainty set) 
'''
def Rbar(th, x, t, y, POL_PROB_1, BASELINE_POL, a_bnd, b_bnd, fq, gamma = 0, WGHTS_ = opt_wrapper, **kwargs ):
    pi_1 = np.asarray(POL_PROB_1(th, x)).flatten()
    t_levels = np.unique(t)
    p_1 = BASELINE_POL(pi_1)
    # old # [lda_opt, wghts, wghts_sum] = find_opt_weights_shorter( np.multiply(y*t, pi_1 - p_1), a_, b_)
    if not hasattr(gamma, "__len__"):
        gammas = gamma * np.ones(len(t_levels))
    else:
        gammas = gamma
    [lda_opt_neg1, wghts_neg1, wghts_sum_neg1] = WGHTS_(gammas[0],np.multiply(-1*y[t == -1], pi_1[t == -1] - p_1[t == -1]), a_bnd[t == -1], b_bnd[t == -1], fq[t == -1])
    [lda_opt_1, wghts_1, wghts_sum_1] = WGHTS_(gammas[1], np.multiply(y[t == 1], pi_1[t == 1] - p_1[t == 1]), a_bnd[t == 1], b_bnd[t == 1], fq[t == 1])
    return lda_opt_1 + lda_opt_neg1

'''
Evaluate loss of policy th under different bounds a_, b_ (corresponding to a different value of Gamma)
PI_1 policy prob: return probability pi(x) = 1
p_1: baseline probability (check in case handed a n x m array) 
a_, b_: bounds corresponding to Gamma being assessed (e.g. Un)
gamma: effective gamma bound (no-op for unbudgeted uncertainty set) 
'''
def Rbar_mt(th, x, t, y, POL_PROB_1, BASELINE_POL, a_bnd, b_bnd, fq, gamma = 0, WGHTS_ = opt_wrapper, **kwargs ):
    pi_1 = np.asarray(POL_PROB_1(th, x, t)).flatten()
    t_levels = np.unique(t)
    p_1 = BASELINE_POL(pi_1, t)
    if len(np.asarray(pi_1).shape) > 1:
        p_1 = [ p_1[i,t[i]] for i in range(x.shape[0]) ]  # project n x m array to a vector based on observed teratment assignment
    # old # [lda_opt, wghts, wghts_sum] = find_opt_weights_shorter( np.multiply(y*t, pi_1 - p_1), a_, b_)
    if not hasattr(gamma, "__len__"):
        gammas = gamma * np.ones(len(t_levels))
    else:
        gammas = gamma
    lda_opt_total = 0
    for ind,t_l in enumerate(t_levels):
        [lda_opt, wghts, wghts_sum] = WGHTS_(gammas[ind], np.multiply(y[t == t_l], pi_1[t == t_l] - p_1[t == t_l]), a_bnd[t == t_l], b_bnd[t == t_l], fq[t == t_l])
        lda_opt_total += lda_opt
    return lda_opt_total



''' vanilla uncentered ipw
extra args: PI_1, x (with intercept), y, t, fq,
'''
def vanilla_ipw(th,*args):
    C = 0.05
    PI_1 = args[0]; x_aug=args[1]; y=args[2]; t=args[3]; fq = args[4]; n = len(y)
    fq_norm = fq/np.sum(fq)
    pi_1 = PI_1(th, x_aug).flatten();
    pi_t = np.asarray( [pi_1[i] if t[i] == 1 else 1- pi_1[i] for i in range(n)] )
    return np.sum( y*pi_t/fq_norm ) + C*np.linalg.norm(th,2)**2
''' vanilla uncentered ipw
extra args: PI_1, x (with intercept), y, t, fq,
'''
def vanilla_ipw_subgrad(th,*args):
    C = 0.05
    PI_1 = args[0]; x_aug=args[1]; y=args[2]; t=args[3]; fq = args[4]; n = len(y)
    fq_norm = fq/np.sum(fq)
    pi_1 = PI_1(th, x_aug).flatten();
    pi_1_grad = qk_dpi_dtheta(pi_1, th, x_aug)
    pi_t_grad = np.asarray( [pi_1_grad[i] if t[i] == 1 else -pi_1_grad[i] for i in range(n)] )
    subgrad = (y/fq_norm).dot(pi_t_grad).T + 2*C*th
    return subgrad

def grad_descent_vanilla_ipw_mt(th, N_RNDS, POL_PROB_1, x, t, fq, y, eta_0,logging=False,step_schedule=0.5):
    n = x.shape[0];
    assert all(len(arr) == n for arr in [x,t,fq,y])
    # If last column is all ones, don't augment
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    risks = np.zeros(N_RNDS); THTS = [None]*N_RNDS; PSTARS = np.zeros([N_RNDS, n]); losses = [None]*N_RNDS; oosrisks = np.zeros(N_RNDS)
    fq_norm = fq/np.sum(fq)
    for k in range(N_RNDS) :
        eta_t = eta_0 * 1.0/np.power((k+1)*1.0, 0.3);
        pi_1 = POL_PROB_1(th, x_aug).flatten();
        pi_t = np.asarray( [pi_1[i] if t[i] == 1 else 1- pi_1[i] for i in range(n)] )
        loss = np.sum( y*pi_t/fq_norm  )
        pi_1_grad = qk_dpi_dtheta(pi_1, th, x_aug)
        pi_t_grad = np.asarray( [pi_1_grad[i] if t[i] == 1 else -pi_1_grad[i] for i in range(n)] )
        subgrad = (y/fq_norm).dot(pi_t_grad).T
        th = th - eta_t * subgrad
        THTS[k] = th; losses[k] = loss
    return [oosrisks, losses, THTS, PSTARS]

# random restarts
def opt_w_restarts_vanilla_ipw_mt(N_RST, th, N_RNDS, POL_PROB_1, x, t, fq, y, eta_0,logging=False,step_schedule=0.5,**params):
    ls = np.zeros(N_RST); ths = [None] *N_RST
    iterator = log_progress(range(N_RST),every=1) if logging else range(N_RST)
    for j in iterator:
        # assume data has intercept
        th_0 = np.random.randn(x.shape[1]);
        [oosrisks, losses, THTS, PSTARS] = grad_descent_vanilla_ipw(th_0, N_RNDS, POL_PROB_1, x, t, fq, y, eta_0,logging=False,step_schedule=step_schedule)
        ls[j]=losses[-1]; ths[j] = THTS[-1]
        if logging:
            plt.plot(range(N_RNDS), losses)
            plt.pause(0.05)
    return [ths[np.argmin(ls)], min(ls)] #return tht achieving min loss


# https://github.com/funsim/moola/blob/master/moola/linesearch/dcsrch_fortran/linesearch.py
# armijo linesearch
class LineSearch:
    """
    A generic linesearch class. Most methods of this
    class should be overridden by subclassing.
    """

    def __init__(self, **kwargs):
        self._id = 'Generic Linesearch'
        return

    def _test(self, func, x, d, slope, f = None, t = 1.0, **kwargs):
        """
        Given a descent direction d for function func at the
        current iterate x, see if the steplength t satisfies
        a specific linesearch condition.
        Must be overridden.
        """
        return True # Must override

    def search(self, func, x, d, slope, f = None, **kwargs):
        """
        Given a descent direction d for function func at the
        current iterate x, compute a steplength t such that
        func(x + t * d) satisfies a linesearch condition
        when compared to func(x). The value of the argument
        slope should be the directional derivative of func in
        the direction d: slope = f'(x;d) < 0. If given, f should
        be the value of func(x). If not given, it will be evaluated.
        func can point to a defined function or be a lambda function.
        For example, in the univariate case::
            test(lambda x: x**2, 2.0, -1, 4.0)
        """
        # return to default stepsize if not a descent dir (due to finite differences and nonconvex opt)
        t = 1.0
        if slope >= 0.0:
            return t
        while not self._test(func, x, d, f = f, t = t, **kwargs):
            pass
        return t


class ArmijoLineSearch(LineSearch):
    """
    An Armijo linesearch with backtracking. This class implements the simple
    Armijo test
    f(x + t * d) <= f(x) + t * beta * f'(x;d)
    where 0 < beta < 1/2 and f'(x;d) is the directional derivative of f in the
    direction d. Note that f'(x;d) < 0 must be true.
    :keywords:
        :beta:      Value of beta (default 0.001)
        :tfactor:   Amount by which to reduce the steplength
                    during the backtracking (default 0.5).
    """

    def __init__(self, **kwargs):
        LineSearch.__init__(self, **kwargs)
        self.beta = max(min(kwargs.get('beta', 1.0e-4), 0.5), 1.0e-10)
        self.tfactor = max(min(kwargs.get('tfactor', 0.1), 0.999), 1.0e-3)
        self.default = kwargs.get('default')
        return

    def _test(self, func, x, d, slope, f = None, t = 1.0, **kwargs):
        """
        Given a descent direction d for function func at the
        current iterate x, see if the steplength t satisfies
        the Armijo linesearch condition.
        """
        if f is None:
            f = func(x)
        f_plus = func(x + t * d)
        return (f_plus <= f + t * self.beta * slope)

    def search(self, func, x, d, slope, f = None, **kwargs):
        """
        Given a descent direction d for function func at the
        current iterate x, compute a steplength t such that
        func(x + t * d) satisfies the Armijo linesearch condition
        when compared to func(x). The value of the argument
        slope should be the directional derivative of func in
        the direction d: slope = f'(x;d) < 0. If given, f should
        be the value of func(x). If not given, it will be evaluated.
        func can point to a defined function or be a lambda function.
        For example, in the univariate case:
            `test(lambda x: x**2, 2.0, -1, 4.0)`
        """
        if f is None:
            f = func(x)

        t = 1.0
        if slope >= 0.0:
            return t # don't linesearch if not a descent direction
        iterations = 0
        while not self._test(func, x, d, slope, f = f, t = t, **kwargs):
            t *= self.tfactor
            # print iterations; sys.stdout.flush()
            iterations += 1
            if iterations >= 20:
                return self.default
                # raise ValueError("line search doesn't terminate")
                # return None
        return t
