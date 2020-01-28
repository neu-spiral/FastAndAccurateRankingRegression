# Fast And Accurate Ranking Regression

The code in this repository implements the algorithms and experiments in the following paper:
> I. Yildiz, J. Dy, D. Erdogmus, J. Kalpathy-Cramer, S. Ostmo, J. P. Campbell, M. F. Chiang, S. Ioannidis, “Fast and Accurate Ranking Regression”, AISTATS, Italy, 2020.

We implement seven inference algorithms to estimate Plackett-Luce scores from choice observations. Four are *feature methods*, i.e., algorithms that regress Plackett-Luce scores from features:  
- PLADMM and PLADMM-log that we propose (implemented in `admm_lin.py` and `admm_log.py`, respectively), 
- Sequential least-squares quadratic programming (SLSQP) that parametrizes scores as linear functions of features as in PLADMM (implemented in `mle_lin.py`), and
- Newton on $\beta$ that parametrizes scores as logistic functions of features as in PLADMM-log and solves the resulting convex problem via Newton's method (implemented in `mle_exp.py`). 

The remaining three are *featureless methods*, i.e., algorithms that learn the Plackett-Luce scores from the choice observations alone (implemented in `only_scores.py`): 
- Iterative Luce Spectral Ranking (ILSR) by Maystre and Grossglauser (2015), 
- Minorization-Maximization (MM) by Hunter (2004), and 
- Newton on $\bm \theta$ that reparametrizes scores via exponentials and solves the resulting convex problem via Newton's method.

We evaluate all algorithms on synthetic and real-life datasets, prepared by the methods in `preprocessing.py`.  We perform 10-fold cross validation (CV) for each dataset. For synthetic datasets, we also repeat experiments over 5 random generations. We partition each dataset into training and test sets in two ways. In *observation CV*, implemented by the *create_partitions* method, we partition the dataset w.r.t. observations, using 90% of the observations for training and the remaining 10% for testing. In *sample CV*, implemented by the *create_partitions_wrt_sample* method, we partition samples using 90% of the samples for training and the remaining 10% for testing. We construct synthetic datasets via *save_synthetic_data*; details of the process are provided in the supplement of the paper. 

We run each algorithm until convergence via `run_methods.py`. We measure the elapsed time, including time spent in initialization, in seconds and the number of iterations. We measure the prediction performance by Top-1 accuracy and Kendall-Tau correlation on the test set. For synthetic datasets, we also measure the quality of convergence by the norm of the difference between estimated and true Plackett-Luce scores. We report averages and standard deviations over folds via the *metric_and_CI* method.

`utils.py` contains fundamental methods reused by all algorithms. *init_params* initializes the Plackett-Luce scores for all seven inference algorithms described above. Particularly, *init_beta_b_convex_QP* initializes the parameter vector for PLADMM, as described in Section 4 of the paper, while *init_exp_beta* initializes the parameter vector for PLADMM-log, as described in the supplement. Given the transition matrix of a Markov-Chain, *statdist* finds the stationary distribution. *objective* evaluates the log-likelihood under the Plackett-Luce model for a given set of scores and choice observations. *check_global_balance_eqn* tests if the given pair of transition matrix and scores satisfy the global balance equations. Given a pair of score estimates and rankings, *top1_test_accuracy* and *kendall_tau_test* evaluate the corresponding average metrics. 

# Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> I. Yildiz, J. Dy, D. Erdogmus, J. Kalpathy-Cramer, S. Ostmo, J. P. Campbell, M. F. Chiang, S. Ioannidis, “Fast and Accurate Ranking Regression”, AISTATS, Italy, 2020.

# Acknowledgements
Our work is supported by NIH (R01EY019474), NSF (SCH-1622542 at MGH; SCH-1622536 at Northeastern; SCH-1622679 at OHSU), and by unrestricted departmental funding from Research to Prevent Blindness (OHSU).
