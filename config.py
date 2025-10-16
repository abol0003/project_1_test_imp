# all configuration variables and constants
import os 

#====================================================
DATA_DIR = r"./data/dataset/"
OUTPUT_PRED = "submission_best.csv"

PICT_DIR = "picture"
CONF_MAT_FIG = os.path.join(PICT_DIR, "confusion_matrix.png")
ROC_FIG      = os.path.join(PICT_DIR, "roc_curve.png")
PR_FIG       = os.path.join(PICT_DIR, "pr_curve.png")

# Paths 
SAVE_DIR = "data_saving"
RAW_DATA = os.path.join(SAVE_DIR, "raw_data.npz")
PREPROC1_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_1.npz")
PREPROC2_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_2.npz")
SAVE_BEST         = os.path.join(SAVE_DIR, "best_params.npz")
SAVE_WEIGHTS      = os.path.join(SAVE_DIR, "final_weights.npy")

#====================================================
# Pipeline 
DO_PREPROCESS = True    # reuse preprocessed npz if False
DO_SUBMISSION = True     # when True: train final model, save weights, build submission & plots
HYPERPARAM_TUNING = True    # tune or load best params

RNG_SEED = 42

#====================================================
# Tuning parameters 
HOLDOUT_VAL_FRAC = 0.15
TUNING_MAX_ITERS = 600
FINAL_MAX_ITERS  = 1000
GAMMA_LOW = 1e-3  
GAMMA_HIGH = 1
LAMBDA_LOW = 1e-12
LAMBDA_HIGH = 1e-5

N_TRIALS = 5  # Number of trials for hyperparameter tuning

#====================================================
# Light One-hot Encoding 
LOW_CARD_MAX_UNIQUE = 15    # Decides if a column should be encoded
ONEHOT_PER_FEAT_MAX = 8     
MAX_ADDED_ONEHOT    = 200   # Limits the total number of new columns added by encoding

#====================================================
# Nature of the features (Preproc)
CAT = [1, 2,  ]      # categorical feature indices
DISC = [5, 8, 15, 22]     # discrete feature indices
CONT = [0, 1, 2, 4, 6]    # continuous feature indices

#====================================================
# Optimizer / Scheduler knobs (tuning + final training)
USE_ADAM_DEFAULT   = True
SCHEDULE_DEFAULT   = "None"  # one of: "none", "cosine", "exponential"
EARLY_STOP_DEFAULT = False
PATIENCE_DEFAULT   = 15
TOL_DEFAULT        = 1e-1
USE_WEIGHTED_BCE   = False  # when True, use class-balanced BCE gradient in training

# Optional search spaces (set >1 values if you want to random search these too)
SCHEDULE_CHOICES = ["cosine"]  # small set by default
ADAM_CHOICES     = [True]

# Polynomial expansion (applied in preprocess2 before standardization)
ADD_POLY_FEATURES        = True
POLY_ADD_SQUARES         = True
POLY_ADD_INTERACTIONS    = True
POLY_MAX_ADDED_FEATURES  = 800
POLY_CONT_MAX_UNIQUE     = 32