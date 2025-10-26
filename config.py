import os

# ==========================
# PATHS & FILES
# ==========================
DATA_DIR = "./data/dataset/"
SAVE_DIR = "data_saving"
PICT_DIR = "picture"

RAW_DATA = os.path.join(SAVE_DIR, "raw_data.npz")
PREPROC1_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_1.npz")
PREPROC2_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_2.npz")
PREPROC3_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_3.npz")
PREPROC4_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_4.npz")
OUTPUT_PRED = "submission_best.csv"
CONF_MAT_FIG = os.path.join(PICT_DIR, "confusion_matrix.png")
ROC_FIG = os.path.join(PICT_DIR, "roc_curve.png")
PR_FIG = os.path.join(PICT_DIR, "pr_curve.png")

SAVE_BEST = os.path.join(SAVE_DIR, "best_params.npz")
SAVE_BEST2 = os.path.join(SAVE_DIR, "best_params_2.npz")
SAVE_WEIGHTS = os.path.join(SAVE_DIR, "final_weights.npy")
SAVE_WEIGHTS_NAGFREE = os.path.join(SAVE_DIR, "final_weights_nagfree.npy")

DO_PREPROCESS = True
DO_SUBMISSION = False
HYPERPARAM_TUNING = True

RNG_SEED = 42


# =========================================================
# PREPROCESSING PARAMETERS
# =========================================================
DROP_FIRST_N_CAT_COLS = 26
LOW_CARD_MAX_UNIQUE = 20
MAX_ADDED_ONEHOT = 2000
ONEHOT_DROP_FIRST = True

PCA_VAR = 0.97
PCA_Local = {"variance_ratio": PCA_VAR, "min_cols": 8, "replace": True}
PCA_K = None
ORDINAL_ENCODE = True
ORDINAL_SCALE_TO_UNIT = True

NAN_INDICATOR_MIN_ABS_CORR = 0.1
NAN_INDICATOR_TOPK = None
NAN_INDICATOR_MIN_PREV = 0.1
NAN_INDICATOR_MAX_PREV = 0.9

IMPUTE_SKEW_RULE = 0.5
IMPUTE_CONT_ALLNAN_FILL = 0.0
IMPUTE_NOM_ALLNAN_FILL = -1.0
IMPUTE_BIN_ALLNAN_FILL = 0.0

STD_CONT = False

POLY_ENABLE_V2 = False
POLY_ADD_SQUARES_CONT = True
POLY_ADD_SQUARES_PCA = False
POLY_ADD_INTER_CONT = False
POLY_ADD_INTER_PCA = False
POLY_ADD_INTER_CROSS = False
POLY_TOPK_PAIRS = 256
POLY_MIN_ABS_CORR = 0.00

PRUNE_CORR_THRESHOLD = 0.90
ADD_BIAS = False


# =========================================================
# HYPERPARAMETER TUNING
# =========================================================
TUNING_MAX_ITERS = 400
NAGFREE_TUNING = True

# GAMMA_LOW = 1e-3
# GAMMA_HIGH = 1.0
LAMBDA_LOW = 1e-8
LAMBDA_HIGH = 1e-5
NAGFREE_L_MAX = 1e8
N_TRIALS = 10

USE_ADAM_DEFAULT = True
SCHEDULE_DEFAULT = "cosine"  # cosine, exponential, none
EARLY_STOP_DEFAULT = True
PATIENCE_DEFAULT = 15
TOL_DEFAULT = 1e-8

USE_WEIGHTED_BCE = True

SCHEDULE_CHOICES = ["exponential, cosine"]  # cosine, exponential, none
ADAM_CHOICES = [True]  # can just be better as it just make converge faster


# =========================================================
# FINAL TRAINING & SUBMISSION
# =========================================================
FINAL_MAX_ITERS = 400
CV_KFOLD_PREDICTION = False
