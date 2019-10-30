import numpy as np
import gurobipy as gp
from unconfoundedness_fns import *
from subgrad import *
from opt_tree import *
from greedy_partitioning_serv import *
import pickle
from datetime import datetime

'''
policy_prob_1: function returning probability of policy assignment = 1
weight_function: function returning adversarial weights (given lower, upper bound and gamma or effective gamma budget bound (L1 budget) 
baseline_pol: pi_0(x) function 
policy_optimizer: ogd, or optimal_tree 
- verbose 
- save 
    - stump 
    - experiment name 
'''
class ConfoundingRobustPolicy:
    def __init__(self, baseline_pol, verbose = False, ind_rep = -1, save_params={}, save = True, treatment_n='binary', params={}):
        self.baseline_pol = baseline_pol
        # self.weight_function = weight_function
        self.verbose = verbose
        self.save_params = save_params #stump, exp_name
        self.save = save
            #self.save_params != {} # check if it's not the default
        self.ind_rep = ind_rep
        self.params = params
        self.treatment_n = treatment_n # some implementations leverage binary encoding

        # store training data when passed in by .fit
        self.x = None
        self.t = None
        self.t_levels = None
        self.y = None
        self.fq = None
        self.RBARS_ = None
        self.RISKS = None
        self.POLS_dict = None
        self.eval_data = None
        self.opt_params_og = None
        self.get_risk = None #overwritten with a frozen test set: call get_risk with a vector of policy prob assignments
    ''' 
    fit based on X, T, Y; q0 nominal propensities
    Assume x doesn't have an intercept
    GAMMAS series of parameters
    method_types: list of strings 'ogd', 'fdiv-02', 
    method_params object takes in 
        - optimizer
        - opt_params
            the params_dict that gets passed to the optimizer
            DEFAULT_POL: the policy to default to if loss is nonnegative
        - pol_opt
            'ogd', 'tree', 'ipw': descriptor of method (gradient based or otherwise) 
                'unc_set_type': 'interval', 'L1-budget'
    # Make GAMS rounded precision
    '''
    def fit(self, x, t, y, q0, GAMS, method_params, eval_conf={'eval':False} ):
        if self.treatment_n == 'binary':
            t = get_sgn_0_1(t) # make sure treatment is signed, not 0-1
        else:
            n_treatments = len(np.unique(t))
        random.seed(1)

        # we input data with intercept
        data_dict = { 'x':x, 't':t, 'y':y, 'fq': q0 }
        self.x = x; self.t = t; self.y = y; self.fq = q0; self.t_levels = np.sort(np.unique(t))
        opt_params = method_params['opt_params']  # get parameters (e.g. optimization parameters from dict)
        self.opt_params_og = opt_params
        opt_params.update({'x': x, 't': t, 'y': y, 'fq': q0})
        if self.treatment_n == 'binary':
            prev_sol = np.random.randn(x.shape[1]);
        else:
            print n_treatments
            print x.shape[1]
            prev_sol = np.random.randn(x.shape[1],n_treatments);
        POLS = [ None ] * len(GAMS);  losses = np.zeros(len(GAMS))
        RISKS = np.zeros(len(GAMS))

        if eval_conf['eval']:  # if evaluate policy error online
            self.init_evaluation(eval_conf)  # set risk_test

        # iterate over values of GAMMAS and train
        for ind_g, gam in enumerate(GAMS):
            if self.verbose:
                print 'gamma, ', gam
            a_bnd, b_bnd = get_bnds(q0, gam)
            data_dict.update({'a_bnd':a_bnd, 'b_bnd':b_bnd}); opt_params.update( {'a_bnd':a_bnd,'b_bnd':b_bnd} )
            # update parameters based on gamma, a_bnd, b_bnd
            optimizer = method_params['optimizer']
            [opt_params, gammas] = self.update_opt_params(gam, a_bnd, b_bnd, opt_params, method_params)
            now = datetime.now()
            [robust_th, robust_loss] = optimizer(th=prev_sol, **opt_params)
            print datetime.now() - now, 'time optimizing'
            # Optimization refinements since gamma uncertainty sets are nested: revert, check loss is negative
            now = datetime.now()
            if method_params['pol_opt'] != 'IPW':
                if ind_g > 1:
                    [robust_th, robust_loss] = self.reversion_check_optimality(ind_g, robust_th, robust_loss,
                            a_bnd, b_bnd, gammas, POLS, POL_PROB_1=opt_params['POL_PROB_1'], WGHTS_ = opt_params['WGHTS_'])
                [robust_th, robust_loss] = self.check_loss_nonnegative(robust_th, robust_loss, opt_params['DEFAULT_POL'])
            prev_sol = robust_th; POLS[ind_g] = robust_th; losses[ind_g] = robust_loss
            print datetime.now() - now, 'time verifying'

            # if evaluate policy error online: report results
            if eval_conf['eval']:
                if self.treatment_n == 'binary':
                    test_rec = opt_params['POL_PROB_1'](robust_th, self.eval_data['x_test'])
                else: # we need treatment information for returning policy probability vector for multiple treatments
                    if eval_conf['eval_type'] == "true_dgp":
                        test_rec = opt_params['POL_PROB_all'](robust_th, self.eval_data['x_test'], self.eval_data['t_test'])
                    else: #condition evaluation on observational treatment assignment in t_test
                        test_rec = opt_params['POL_PROB_1'](robust_th, self.eval_data['x_test'], self.eval_data['t_test'])

                RISKS[ind_g] = self.get_risk(test_rec)
                self.RISKS = RISKS
                if self.verbose:
                    print 'eval risk', RISKS[ind_g]
            if self.verbose:
                pickle.dump({'recs': test_rec, 'risk':RISKS[ind_g]}, open(self.save_params['stump'] + self.save_params['exp_name'] + '--' + method_params[
                        'type'] + '--gamma--'+ str(np.round(gam,2)) + '--rep' + str(self.ind_rep) + '--' + datetime.now().strftime(
                        '%Y-%m-%d-%H-%M-%S') + '.p', 'wb'))

        # After fitting: class contains a list of policies
        # Post processing of optimization results; calibrate,
        if method_params['pol_opt'] != 'IPW':
            self.RBARS_ = self.calibrate_risk_for_policies(GAMS, POLS, unc_set_type=method_params['unc_set_type'], POL_PROB_1=opt_params['POL_PROB_1'], WGHTS_=opt_params['WGHTS_'])
        self.POLS_dict = dict(zip(GAMS, POLS))
        if self.save:
            pickle.dump( {'RBARS_':self.RBARS_, 'POLS_dict': self.POLS_dict, 'GAMS':GAMS, 'RISKS':RISKS}, open(self.save_params['stump']+self.save_params['exp_name']+'--'+method_params['type']+'--rep'+str(self.ind_rep)+'--'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.p', 'wb') )

    def predict(self, x, gamma):
        pol_fn = POLS_dict[str(gamma)]
        return opt_params['POL_PROB_1'](pol_fn, x)
            # self.POLS_dict[str(gamma)](x)


    ''' update opt params for changes in gamma, a_bnd, b_bnd computed 
    '''
    def update_opt_params(self, gamma, a_bnd, b_bnd, opt_params, method_params):
        gammas = gamma
        if method_params['pol_opt'] == 'ogd':
            [N_RNDS, eta, N_RNDS_tv] = self.get_ogd_params(a_bnd, b_bnd)
            if method_params['unc_set_type'] == 'interval':
                opt_params.update({'N_RNDS': N_RNDS, 'gamma':gammas, 'step_schedule': eta, 'eta_0': 1})
            elif method_params['unc_set_type'] == 'L1-budget':
                fq_sums = [ (1.0 / self.fq[self.t == t_]).sum() for t_ in self.t_levels ]
                fq_nm_wghts = [(1 / self.fq[self.t == self.t_levels[ind]]) / fq_sums[ind] for ind in range(len(self.t_levels)) ]
                upper_gamma_bnds = np.asarray([np.sum(np.maximum(a_bnd[self.t == t_] / fq_sums[ind] - fq_nm_wghts[ind], b_bnd[self.t == t_] / fq_sums[ind] - fq_nm_wghts[ind])) for ind,t_ in enumerate(self.t_levels)])
                gammas = opt_params['rho']*upper_gamma_bnds
                opt_params.update({'N_RNDS': N_RNDS_tv, 'gamma': gamma, 'step_schedule': eta, 'eta_0': 1})
        elif method_params['pol_opt'] == 'tree':
            None
        elif method_params['pol_opt'] == 'IPW':
            opt_params.update({'eta_0': 1})
            [psub_0_th_ipw, ls_th_ipw] = opt_w_restarts_vanilla_ipw(th = 0, **opt_params)
                                                            # N_RNDS, PI_1, x_[:, 0:d], t_, nominal_Q_, y_, 1,logging=False, step_schedule=IPW_STEP)
        return [opt_params, gammas]

    def get_upper_gamma_bnds(self, a_bnd, b_bnd):
        fq_sums = [(1.0 / self.fq[self.t == t_]).sum() for t_ in self.t_levels]
        fq_nm_wghts = [(1 / self.fq[self.t == self.t_levels[ind]]) / fq_sums[ind] for ind in range(len(self.t_levels))]
        upper_gamma_bnds = np.asarray([np.sum(np.maximum(a_bnd[self.t == t_] / fq_sums[ind] - fq_nm_wghts[ind],
                                                         b_bnd[self.t == t_] / fq_sums[ind] - fq_nm_wghts[ind])) for
                                       ind, t_ in enumerate(self.t_levels)])
        gammas = self.opt_params_og['rho'] * upper_gamma_bnds
        return gammas

    ''' 
    'eval_type': assume known dgp or RCT evaluation 
    '''
    def init_evaluation(self, eval_conf):
        eval_type = eval_conf['eval_type']
        self.eval_data = eval_conf['eval_data']
        if eval_type == "true_dgp":
            oracle_risk = eval_conf['oracle_risk']
            def get_risk(pi_test, baseline_pol = self.baseline_pol):
                x_test = self.eval_data['x_test']; u_test = self.eval_data['u_test']; t_test = self.eval_data['t_test']
                if self.treatment_n == 'binary':
                    return np.mean(oracle_risk(pi_test, x_test, u_test) - oracle_risk( baseline_pol(x_test) , x_test, u_test))
                else:
                    # For evaluating multiple treatments with oracle outcomes, compute
                    # \sum_i \sum_t   Y_i(t) (\pi(t, X_i) - \pi_0(t, X_i) )
                    baseline_assignment = np.zeros(pi_test.shape)
                    for t in np.unique(t_test):
                        baseline_assignment[:, t] = baseline_pol(x_test, t*np.ones(x_test.shape[0]))
                    return oracle_risk(pi_test, x_test, u_test) - oracle_risk(baseline_assignment, x_test, u_test)
            self.get_risk = get_risk

        elif eval_type == "rct":  # if evaluate on RCT data
            def get_risk(pi_test, baseline_pol = self.baseline_pol):
                x_test = self.eval_data['x_test']
                t_test = self.eval_data['t_test']
                y_test = self.eval_data['y_test']
                pi_test = np.asarray(pi_test)
                RCT_prop_test = self.eval_data['RCT_q0'] # propensities from RCT

                if sum(pi_test) > 0:  # if treat at all
                    if self.treatment_n == 'binary':
                        t_test_sgn = get_sgn_0_1(t_test)  # if evaluating on an rct: need to put this in fulldata dict
                        return np.mean( y_test * (pi_test - baseline_pol(x_test)) * t_test_sgn / RCT_prop_test )
                    else:
                        # \sum_i  Y_i (\pi(T[i]) - \pi_0(T[i]) ) / Q[i,T[i]]
                        return np.mean(y_test * (pi_test - baseline_pol(x_test, t_test)) / RCT_prop_test)
                else:
                    return 0 # assuming baseline is all-control
            self.get_risk = get_risk

    def calibrate_risk_for_policies(self, GAMS, POLS, unc_set_type, POL_PROB_1, WGHTS_ = opt_wrapper):
        ''' Compute a calibration plot under "GAMS" gamma parameters for "POLS" policies

        calibration assesses the worst case regret of a policy learned under an assumption of Gamma = Gamma'
        which would be incurred were the true state of the world Gamma_calib
        '''
        RBARS_ = np.zeros([len(GAMS),len(GAMS)])
        # Post processing, Rbar # compute policy unconfounded  # data_dict: x_, t_sgned_, y_, PI_1, PREF, fq, WGHTS_
        for ind_gam_k in range(len(GAMS)):
            for ind_g_calib, gam_calib in enumerate(GAMS):
                a_calib, b_calib = get_bnds(self.fq, gam_calib)                 # previously was used to calib for IST data # a_calib, b_calib = get_bnds(q0[train_ind]*nominal_selection_[train_ind], gam_calib)
                if unc_set_type == 'L1-budget':
                    gammas = self.get_upper_gamma_bnds(a_calib, b_calib)
                    gamma = gammas
                else:
                    gamma = 0
                if self.treatment_n == 'binary':
                    RBARS_[ind_gam_k, ind_g_calib] = Rbar(th = POLS[ind_gam_k],
                                                      x=self.x, t=self.t, y=self.y, POL_PROB_1 = POL_PROB_1,BASELINE_POL= self.baseline_pol,
                                                      a_bnd = a_calib, b_bnd = b_calib, fq=self.fq, gamma = gamma,WGHTS_ = WGHTS_)
                else:
                    RBARS_[ind_gam_k, ind_g_calib] = Rbar_mt(th = POLS[ind_gam_k],
                                                      x=self.x, t=self.t, y=self.y, POL_PROB_1 = POL_PROB_1,BASELINE_POL= self.baseline_pol,
                                                      a_bnd = a_calib, b_bnd = b_calib, fq=self.fq, gamma = gamma,WGHTS_ = WGHTS_)

        self.RBARS_ = RBARS_
        return RBARS_
    '''
    Check losses on previous policies 
    # TODO: fix 
    '''
    def reversion_check_optimality(self, ind_g, th, ls_th, a_bnd, b_bnd, gammas, POLS, POL_PROB_1, WGHTS_ = opt_wrapper):
    #### Reversion for non-fdiv policies
        RBARS_opt = np.zeros(ind_g)

        for ind_gam_i in range(ind_g):  # for every gamma_i < gamma_k
            # Evaluate previously found policies on current data; with current Gamma parameter bounds
            if self.treatment_n=='binary':
                RBARS_opt[ind_gam_i] = Rbar(th = POLS[ind_gam_i],
                                        x=self.x, t=self.t, y=self.y, POL_PROB_1 = POL_PROB_1, BASELINE_POL= self.baseline_pol,
                                        a_bnd = a_bnd, b_bnd = b_bnd, fq = self.fq, gamma = gammas, WGHTS_ = WGHTS_)
            else:
                RBARS_opt[ind_gam_i] = Rbar_mt(th = POLS[ind_gam_i],
                                        x=self.x, t=self.t, y=self.y, POL_PROB_1 = POL_PROB_1, BASELINE_POL= self.baseline_pol,
                                        a_bnd = a_bnd, b_bnd = b_bnd, fq = self.fq, gamma = gammas, WGHTS_ = WGHTS_)
        if self.verbose:
            print RBARS_opt, 'others evaled'
            print ls_th, 'current loss'
        if np.min(RBARS_opt) < ls_th:  # if there is a policy achieving better risk, evaluated on this gamma
            th = POLS[np.argmin(RBARS_opt)]  # set the policy to the one achieving the minimum evaluation
            if self.verbose:
                print 'reverting at ' + str(ind_g) + 'to policy learned at ' + str(np.argmin(RBARS_opt))
        return [th, ls_th]

    ''' revert to control policy (for linear policy, assuming intercept is last coefficient)'''
    def check_loss_nonnegative(self, robust_th, ls, baseline_th, TOL = 1e-4):
        if ls > TOL:
            robust_th = baseline_th
            if self.verbose:
                print 'truncating because nonnegative loss'
            ls = 0 # truncate loss
        else:
            robust_th = robust_th
        return [robust_th, ls]



    ''' get parameter values for OGD (perhaps clipping) 
    '''
    def get_ogd_params(self, a_bnd, b_bnd):
        D = np.linalg.norm((a_bnd - b_bnd) / sum(a_bnd))
        G = np.linalg.norm(0.25 * max(np.abs(self.y)) * self.x.shape[1])  # assume we bound the 2-norm of $\Theta$ by p
        eps = 0.05
        N_RNDS = np.clip(int(G ** 2 * D ** 2 / eps ** 2), 50, 200)
        eta = np.clip((D * G) / np.sqrt(N_RNDS), 0.5, 0.6)
        N_RNDS_tv = np.clip(N_RNDS, 10, 20)  # use OGD parameters
        return [N_RNDS, eta, N_RNDS_tv]



''' opt tree wrapper calls
X__: preprocessed training data (standardize train/test data with same )
t (unsigned)
Y
D (depth)
nominal_Q_,
pi_0,
a_bnd_train,
b_bnd_train
Pass a signed treatment vector to the greedy warm start; but use the sharp version of optimal tree
which uses integer encoding of t
'''
def get_opt_tree_policy(BASELINE_POL, x, t, fq, y, depth, a_bnd, b_bnd, sharp = True, verbose = False,TIME_LIMIT = 180, **params):
    # optimal tree globals
    # remove intercept
    x, eps = preprocess_data_oct_pol(x[:,:-1])
    n = len(y); n_ts = len(np.unique(t))
    pi_0 = BASELINE_POL(x, t)
    K=2; N = 2**(depth+1) - 1; N_nodes = np.arange(2**(depth+1) - 1)+1; N_TB = int(np.floor(N/2))
    branch_nodes = N_nodes[0:(N_TB )]; N_L = N - N_TB
    leaf_nodes = N_nodes[N_TB:]
    A_L = [[] for i in range(N)]; A_R = [[] for i in range(N)]
    [A_L, A_R] = get_ancestors(N_nodes, A_L, A_R)
    [lowers, uppers] = getterminalindices(N)
    #### train on optimal tree
    T_sgned = get_sgn_0_1(t); # this makes a naive binarization
    y_label = 0
    mode_y = mode(y)[0][0]
    L_hat = len(np.where(y == mode_y)[0])*1.0/ len(y)
    [n, dim_d] = x.shape
    [tree, warm_a, warm_b, warm_d, warm_z, warm_c ] = greedy_ws(N_TB, N_L, x,
            y, T_sgned, pi_0, a_bnd, b_bnd, depth, leaf_nodes)
    warm_c = np.ones(N_L)*0.5
    # get leaf labels from warm start
    leaf_labels = np.ones(N_L)*0.5
    [warm_leaves,leaf_cts] = causal_tree_pred(x, leaf_nodes, warm_a, warm_b, leaf_labels, leaf_lbl=False)
    # fill warm_z
    for i in range(n):
        which_leaf = int(warm_leaves[i])-leaf_nodes[0]
        warm_z[i,which_leaf] = 1
    if verbose:
        print 'ws: ', warm_a, warm_b, warm_d
        print warm_z.sum(axis=0), 'start leaf node counts'
    [m_pol,a_pol,b_pol,d_pol, z_pol, c_pol, l_pol, P, lmbda, u, v] = policy_opt_tree_centered_on_p_0(x,
    y,t,fq, pi_0, a_bnd, b_bnd, A_L, A_R, lowers, uppers, leaf_nodes,
    branch_nodes, L_hat, N, K, eps, y_label, warm_a, warm_b, warm_d, warm_z, warm_c, sharp = sharp, TIME_LIMIT= TIME_LIMIT)
    lmbda = lmbda.X
    a_pol_ = np.zeros([dim_d, N_TB])
    for i in range(N_TB):
        a_pol_[:,i] = np.asarray([ a_pol[p,i+1].X for p in range(dim_d) ]).T
    b_pol_ = [ b_pol[i+1].X for i in range(N_TB) ]
    leaf_labels = np.asarray([ [c_pol[t_l, t].X for t in range(n_ts)  ]  for t_l in leaf_nodes ])
#         leaf_labels = [ 0 if c_pol[t_l].X > 0 else 1 for t_l in leaf_nodes ]
    if verbose:
        print a_pol_, b_pol_
        print ' c assignment: ', [c_pol[t_l,t].X for t in range(n_ts)  for t_l in leaf_nodes ]
        print ' P assignment: ', [P[i,t].X  for t in range(n_ts)  for i in range(n) ]
        print leaf_labels
        print 'leaf nonempty', [l_pol[t_l].X for t_l in leaf_nodes ]
        if sum(b_pol_) < 0.001:
            print 'empty solution'

    # pickle.dump( {'a': a_pol_,'b': b_pol_, 'leaves': leaf_labels, 'ws-tree': tree}, open('data/out/syn-glb/CR/opt-tree-gam-'+str(gam)+'-rep-'+str(ind_rep+5) + '-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.p', 'wb'))
    # [Y_pred, leaf_node_cts] = causal_tree_pred(x_test, leaf_nodes, a_pol_, b_pol_,leaf_labels)
    # print leaf_node_cts, 'leaf node cts'
    # tree_rec = [tree.get_assignment(x_test[i,:]) for i in range(xtest.shape[0]) ]
    # print 'sum Y pred: ' + str(sum(Y_pred))
    # print 'sum treerec: ' + str(sum(tree_rec))
    tree_pol = {'a': a_pol_,'b': b_pol_, 'leaves': leaf_labels, 'leaf_nodes': leaf_nodes}
    return [tree_pol, lmbda]