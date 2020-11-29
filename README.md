# Code for Minimax Policy Learning under Unobserved Confounding 



```ConfoundingRobustPolicy``` is a wrapper object. The main choices are: 
- Optimizer
  - Gradient-based approaches
  - Global optimization
- Uncertainty set and weight subproblem solution
  - This is passed to the gradient-based approach as a function callback for computing the optimal weights. 
- Performance Evaluation
  - simulation (known dgp) 
  - observational data (pass in propensity scores, evaluate by IPW) 
- Type of treatment
  - binary treatment (we reparametrize with respect to the probability of assigning ```T=1```)
  - multiple treatments
  
  
The ```.fit()``` method is specialized to handle various combinations of the above configurations. 

```ConfoundingRobustPolicy``` takes as input: 
- ```baseline_pol``` (function returning baseline policy) 


```ConfoundingRobustPolicy.fit()``` takes as input: 
- ```X``` data
- ```T``` data (integer-coded) 
- ```Y```
- ```q0``` nominal propensities
- ```log gamma``` series of sensitivity parameters to optimize over (some approaches leverage the nested structure of uncertainty sets
- ```optimization params``` a dictionary of optimization parameters
- ```eval_conf``` a dictionary of evaluation parameters. 

 ```optimization params``` is a dictionary with the following configuration parameters: 
 
-```optimizer```: function callback, e.g. ```get_opt_tree_policy```.
-```pol_opt```: name of policy class 
- ```unc_set_type```: indicator of uncertainty set type (interval or budgeted) 
- ```opt_params```: method-specific parameters (e.g. step size for gradients; tree depth for optimal tree) 
- ```BASELINE_POL```: ctrl_p_1_mt, 
-```type```:'IPW'

Performance Evaluation 
- For multiple treatments: ```oracle_risk``` is a function that takes in the ``` n x k ``` matrix of policy assignment probabilities (robust, and baseline), integer-coded treatments. ```oracle_risk``` sums over treatment partitions. 
  
** Replication code **
Run methods_test.py to replicate simulation from paper. 

