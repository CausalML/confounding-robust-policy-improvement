from unconfoundedness_fns import *
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import random
import sys
from subgrad import *
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


from datetime import datetime

# Parameters:
# gamma (comma separated string)
# lambda
# subsample proportion
STUMP = 'WHI-OGD-armijo-morereps-final-calibrated'
arr = sys.argv[1].split(',')
GAMS = [float(gam) for gam in arr]
lmbda = float(sys.argv[2])
DESCRIPTION = sys.argv[3] # description of this run
subsampled = False
EXPNAME = 'OGD-armijo-rs_noprev_uncentered-reps-custometa'
print(GAMS), 'gammas'
print(lmbda), 'lmbda'
N_REPS = 30

random.seed(2)

''' helper methods'''
def get_outcome_models(train_dict):
    regr = GradientBoostingRegressor(max_depth=4, random_state=0, n_estimators=500)
    regr.fit(train_dict['X'], train_dict['Y'])
    pred = regr.predict(train_dict['X'])
    print 'score R2', regr.score(train_dict['X'],train_dict['Y'])
    return [pred, regr]

'''
Configure methods
'''
PI_1 = logistic_pol_asgn # fn returning Pr[pi(x) = 1]
TV_WGHTS = get_general_interval_wghts_algo_centered_TV_prob
GRAD_ = get_implicit_grad_centered_anti_pi
WGHTS_ = opt_wrapper
GRAD_CTR = get_implicit_grad_centered #control or treatment
PI_1 = logistic_pol_asgn # fn returning Pr[pi(x) = 1]
POL_GRAD = qk_dpi_dtheta
PREF = ctrl_p_1

'''
Read data
'''
def read_data_risks(train_dict, test_dict, subsample_dict, ind, GAMS):
    labels = ['ipw', 'log']
    train_ind = subsample_dict['train_ind'] # indexes into observational study data
    test_ind = subsample_dict['test_ind'] # indexes into clinical trial data
    # Load data
    Y_test_full = test_dict['Y']; T_test_full = test_dict['T']; x_test_full = test_dict['X']
    y_test = Y_test_full[test_ind]; t_test = T_test_full[test_ind]; x_test = x_test_full[test_ind,:]
    Y_train_full = train_dict['Y']; T_train_full = train_dict['T']; x_train_full = train_dict['X']
    Z_train_full = train_dict['Z']
    y = Y_train_full[train_ind]; t = T_train_full[train_ind];
    x = x_train_full[train_ind,:]; z = Z_train_full[train_ind]
    d = x_train_full.shape[1]-1
    nominal_Q = train_dict['prop_T'][train_ind]
    x_test_aug = np.hstack([x_test, np.ones([x_test.shape[0],1])])
    # all get the probability of being selected in
    renormalize_theta = lambda tht: tht / np.linalg.norm(tht)
    print len(y_test); print len(t_test)
    n_pols = 2
    RISKS_ = np.zeros([len(GAMS), n_pols]); POLS = [[] for i in range(len(GAMS))]
    prev_sols = [ np.random.randn(x.shape[1]) for i in range(len(labels)) ]
    for ind_gam, Gamma in enumerate(GAMS):
        print Gamma, 'gamma'
        a_bnd, b_bnd = get_bnds(nominal_Q, Gamma)
        q_l = 1/b_bnd; q_h = 1/a_bnd
        p_0 = np.zeros(x.shape[0]);
        def get_oos_est_anti(pi_test):
            pi_test = np.asarray(pi_test) #broadcast
            q_0 = 1.0/2; q_1 = 1.0/2
            t_test_sgn = get_sgn_0_1(t_test)
            q_t_test = 0.5*np.ones(len(pi_test))
            if sum(pi_test)>0: # if treat at all
                return np.mean(y_test*pi_test*t_test_sgn)
            else:
                return 0 # no regret against assigning control
        t_sgned = get_sgn_0_1(t)

        th = np.random.randn(x.shape[1])*0.5;
        th_ctrl = np.zeros(x.shape[1]);
        th_ctrl[-1] = -np.inf
        N_RNDS_tv=10; N_RNDS = 100
        D = np.linalg.norm((a_bnd - b_bnd)/sum(a_bnd) ); G = np.linalg.norm(0.25*max(np.abs(y))*x_test.shape[1]) # assume we bound the 2-norm of $\Theta$ by p
        eps=0.05
        N_RNDS = np.clip(int(G**2*D**2/eps**2),50,400); eta = np.clip((D*G)/np.sqrt(N_RNDS), 0.3,0.5)
        x_aug = np.hstack([x, np.ones([x.shape[0],1])])
        print eta, 'eta', N_RNDS, 'n-rnds'
        then = datetime.now()
        [opt_ipw,ls_th_ipw] = opt_w_restarts_vanilla_ipw(10, th, N_RNDS, PI_1, x[:,0:d],
        t_sgned, nominal_Q, y , 2,logging=False,step_schedule=eta)
        print datetime.now() - then, 'done with ipw'; then = datetime.now()
        opt_ipw_w = th_ctrl; ls_th = 0
        # [opt_ipw_w, ls_th] = opt_w_restarts(10, prev_sols[1], N_RNDS, WGHTS_,
        #         GRAD_CTR, POL_GRAD, PI_1, PREF, x[:,0:d], t_sgned, nominal_Q, y,
        #         a_bnd, b_bnd, Gamma,2,logging=False,step_schedule=eta,averaging=True,give_initial=True)
        # print datetime.now() - then
        POLS[ind_gam] = [opt_ipw, opt_ipw_w]

        print ls_th_ipw, 'loss: opt confounded ipw'
        print ls_th, 'loss: opt robust ipw'
        print datetime.now()-then; then = datetime.now()

        if ind_gam > 1:
            RBARS_ = np.zeros(ind_gam)
            for ind_gam_i in range(ind_gam): # for every gamma_i < gamma_k
                RBARS_[ind_gam_i] = Rbar(POLS[ind_gam_i][1], x, t_sgned, y, PI_1, PREF, a_bnd, b_bnd)
            print RBARS_, 'others evaled'
            print ls_th, 'current loss'
            if np.min(RBARS_) < ls_th: # if there is a policy achieving better risk, evaluated on this gamma
                opt_ipw_w = POLS[np.argmin(RBARS_)][1] # set the policy to the one achieving the minimum evaluation
                print 'reverting at ' + str(ind_gam) + 'to policy learned at ' + str(np.argmin(RBARS_))

        opt_ipw_w_og = opt_ipw_w
        if ls_th > 1e-4:
            print 'truncating log'
            opt_ipw_w = th_ctrl
        else:
            opt_ipw_w = opt_ipw_w

        recs = [PI_1(opt_ipw, x_test_aug), PI_1(opt_ipw_w, x_test_aug)] #, PI_1(fdiv_th, x_test_aug)]
        risks = np.asarray([get_oos_est_anti(recs[i]) for i in range(len(recs))])
        print 'conf ipw ttd %', sum(PI_1(opt_ipw, x_test_aug))/x_test_aug.shape[0]
        print 'conf-robust ipw ttd %', sum(PI_1(opt_ipw_w, x_test_aug))/x_test_aug.shape[0]
        prev_sols = POLS[ind_gam]

        filename_data = 'WHI-case-study/out/out-'+STUMP+'/'+EXPNAME+'--pols-'+ str(Gamma)+ '--lmbda-' + str(lmbda) + '--rep-' + str(ind) +'--'+ datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pkl'
        pickle.dump( { 'opw_ipw_w': opt_ipw_w,'ls_th':ls_th,'opt_ipw_w_og':opt_ipw_w_og, 'opt_ipw':opt_ipw} ,  open( filename_data, "wb" ) )
        filename_recs = 'WHI-case-study/out/out-'+STUMP+'/'+EXPNAME+'--recs-'+ str(Gamma)+ '--lmbda-' + str(lmbda) + '--rep-' + str(ind) +'--'+ datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pkl'
        pickle.dump( recs , open( filename_recs, "wb" ) )
        print risks
        RISKS_[ind_gam,:] = risks
        fn = 'WHI-case-study/out/out-'+STUMP+'/'+EXPNAME+'--risks--gam-' + str(Gamma) + '--lmbda-' + str(lmbda) + '--rep-' + str(ind) +'--'+ datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pkl'
        pickle.dump( risks , open( fn, "wb" ) )


    return RISKS_

def get_risks_from_dicts(train_dict, test_dict, subsample_dicts, ind, GAMS, labels):
    RISKS = np.zeros([len(subsample_dicts), len(labels)])
    for i in range( len(subsample_dicts)):
        print subsample_dicts[i]
        subsample_dict_ = pickle.load(open( subsample_dicts[i], "rb" ))
        risks = read_data_risks(train_dict, test_dict, subsample_dict_, i, GAMS)
        RISKS[i,:] = risks
        fn = 'data/out/'+STUMP+'/risks--rep-' + str(i) +'--'+ datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pkl'
        pickle.dump( risks , open( fn, "wb" ) )
        print risks
    return RISKS

# Load the data_dicts and run and output forests for every gamma

datetime.now()
train_dict = pickle.load(open('WHI-case-study/WHI_OS.pkl'))
# train_dict = pickle.load(open('WHI-case-study/WHI_OS-subsampled.pkl')) # testing
sysbpOS = train_dict['Y']
# Center outcomes
[pred_T_train, regr] = get_outcome_models(train_dict)
centered_constte_shift_train = (sysbpOS-pred_T_train)
# train_dict['Y'] = centered_constte_shift_train + lmbda *train_dict['Yhf'] #frs_bp + lmbda *train_dict['Yhf']/3.0
train_dict['Y'] = centered_constte_shift_train + lmbda*(train_dict['T'].astype(int))
# train_dict['Y'] = sysbpOS + lmbda *train_dict['Yhf'] #frs_bp + lmbda *train_dict['Yhf']/3.0
test_dict = pickle.load(open('WHI-case-study/WHI_CT.pkl'))
test_dict['Y'] = test_dict['Y']+ lmbda*test_dict['T'].astype(int)
# test_dict['Y'] = test_dict['Y'] + lmbda *test_dict['Yhf'] #frs_bp + lmbda *test_dict['Yhf']/3.0
############


RCT_DIFF_MEANS = np.mean(test_dict['Y'][test_dict['T']==1]) - np.mean(test_dict['Y'][test_dict['T']==0]), 'rct diff means estimate for TE'
print RCT_DIFF_MEANS, 'rct diff means estimate'
#train_ind = np.random.choice(len(train_dict['Y']) , size = int(np.round(subsample_train_prop*len(train_dict['Y']) )), replace = True )
if not subsampled:
    subsample_dict = { 'train_ind':range(len(train_dict['Y'])), 'test_ind':range(len(test_dict['Y'])) }
ind = 0

# Scale data
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
# add intercept
train_dict['X'] = np.hstack([train_dict['X'], np.ones([train_dict['X'].shape[0],1])])

labels = [ 'ipw', 'robust']
RISKS__ = [None] * N_REPS
if not subsampled:
    for n_rep in range(N_REPS):
        RISKS_ = read_data_risks(train_dict, test_dict, subsample_dict, ind, GAMS)
        print RISKS_
        print RISKS_[:,0].mean(), 'mean ipw'
        pickle.dump({ 'desc':DESCRIPTION, 'rct-diff-means':RCT_DIFF_MEANS, 'risks': RISKS_, 'gam':GAMS}, open('RISKS_'+EXPNAME+'_adj_pols-lmbda-'+str(lmbda)+'-rep-'+ str(n_rep) +'.pkl', 'wb'))
        RISKS__[n_rep] = RISKS_
    pickle.dump({'risks-all':RISKS__, 'desc':DESCRIPTION}, open('RISKS_ALL-'+EXPNAME+'_adj_pols_'+str(lmbda)+'.pkl', 'wb'))
else: # if subsampled,
    print "subsampling"
    subsample_dicts = glob.glob("WHI-case-study/WHI-train-ind*.pkl")
    RISKS_ = get_risks_from_dicts(train_dict, test_dict, subsample_dicts, ind, GAMS, labels)
    pickle.dump({'risks': RISKS_, 'gam':GAMS,'fm-files':subsample_dicts}, open('RISKS_'+EXPNAME+'_adj_pols_'+str(lmbda)+'.pkl', 'wb'))
