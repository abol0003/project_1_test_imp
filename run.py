"""
CS-433 Project 1 pipeline: preprocess → tune (CV) → train final → submit.
- Windows-safe multiprocessing via spawn.
- Caches: preprocessed arrays (npz), best CV params, final weights.
"""
import os
import time
import numpy as np
import multiprocessing as mp

import helpers
import implementations
import config
import preprocessing
import metrics
import cv_utils


# Ensure output dir exists and set RNG seed once for reproducibility
os.makedirs(config.SAVE_DIR, exist_ok=True)
np.random.seed(config.RNG_SEED)


def sample_loguniform(low: float, high: float, size: int, rng=np.random.RandomState(config.RNG_SEED)) -> np.ndarray:
    """Sample `size` values from a log-uniform distribution over [low, high]."""
    lo, hi = np.log(low), np.log(high)
    return np.exp(rng.uniform(lo, hi, size))



def tune_hyperparameter(X_tr, y_tr_01, folds):
    """Random-search λ, γ (and optimizer/schedule choices) with stratified K-fold CV.
    Returns: best_lambda, best_gamma, best_thr, best_adam, best_schedule_name
    """
    t_tune = time.time()

    if config.HYPERPARAM_TUNING:
        # Sample λ and γ log-uniformly
        lambda_samples = sample_loguniform(config.LAMBDA_LOW, config.LAMBDA_HIGH, config.N_TRIALS)
        gamma_samples = sample_loguniform(config.GAMMA_LOW, config.GAMMA_HIGH, config.N_TRIALS)

        # Build CV tasks, sampling optimizer and schedule choices if provided
        tasks = [
            (
                y_tr_01,
                X_tr,
                folds,
                lam,
                gam,
                config.TUNING_MAX_ITERS,
                np.random.choice(config.ADAM_CHOICES),
                np.random.choice(config.SCHEDULE_CHOICES),
                config.EARLY_STOP_DEFAULT,
                config.PATIENCE_DEFAULT,
                config.TOL_DEFAULT,
            )
            for lam, gam in zip(lambda_samples, gamma_samples)
        ]

        # Windows-friendly spawn context
        nproc = max(1, (os.cpu_count() or 2) - 4)
        with mp.get_context("spawn").Pool(processes=nproc) as pool:
            results = pool.map(cv_utils.cv_train_and_eval, tasks)

        # Select best by F1, keeping the chosen adam/schedule for the winning trial
        best_pack = None
        best_score = -np.inf
        for idx, res in enumerate(results):
            lam, gam, thr, acc, prec, rec, f1 = res
            adam_choice = tasks[idx][6]
            sched_choice = tasks[idx][7]
            crit = f1
            if crit > best_score:
                best_score = crit
                best_pack = (lam, gam, thr, acc, prec, rec, f1, adam_choice, sched_choice)

        best_lambda, best_gamma, best_thr, val_acc, val_prec, val_rec, val_f1, best_adam, best_sched = best_pack
        print(
            f"[BEST-CV] lambda={best_lambda:.3e}, gamma={best_gamma:.3e}, thr={best_thr:.3f}, "
            f"ACC={val_acc:.4f}, P={val_prec:.4f}, R={val_rec:.4f}, F1={val_f1:.4f}, "
            f"adam={best_adam}, schedule={best_sched}"
        )

        # Persist best params (+ extras for final training)
        np.savez(
            config.SAVE_BEST,
            lambda_=best_lambda,
            gamma=best_gamma,
            thr=best_thr,
            acc=val_acc,
            prec=val_prec,
            rec=val_rec,
            f1=val_f1,
            adam=best_adam,
            schedule=best_sched,
        )
        print(f"[Saved] Best params -> {config.SAVE_BEST}")

    else:
        if not os.path.exists(config.SAVE_BEST):
            raise FileNotFoundError(f"{config.SAVE_BEST} not found.")
        npz = np.load(config.SAVE_BEST, allow_pickle=False)
        best_lambda = float(npz["lambda_"])
        best_gamma = float(npz["gamma"])
        best_thr = float(npz["thr"])
        val_acc = float(npz["acc"])
        val_prec = float(npz["prec"])
        val_rec = float(npz["rec"])
        val_f1 = float(npz["f1"])
        best_adam = bool(npz["adam"]) if "adam" in npz.files else None
        best_sched = str(npz["schedule"]) if "schedule" in npz.files else None
        print(f"[Loaded] Best params from -> {config.SAVE_BEST}")
        print(
            f"[BEST] lambda={best_lambda:.3e}, gamma={best_gamma:.3e}, thr={best_thr:.3f}, "
            f"ACC={val_acc:.4f}, P={val_prec:.4f}, R={val_rec:.4f}, F1={val_f1:.4f}, "
            f"adam={best_adam}, schedule={best_sched}"
        )

    print(f"[Hyperparameter Tuning] {time.time() - t_tune:.1f}s")
    return best_lambda, best_gamma, best_thr, best_adam, best_sched


def train_final_model(X_tr, y_tr_01, best_lambda, best_gamma, use_adam=None, schedule_name=None):
    """Train reg-logistic on full train with best params; save weights; return w."""
    t_final = time.time()

    w0 = np.zeros(X_tr.shape[1], dtype=np.float32)
    # Decide optimizer/schedule for final training:
    final_use_adam = use_adam if use_adam is not None else getattr(config, "USE_ADAM_DEFAULT", True)
    final_sched_name = schedule_name if schedule_name is not None else getattr(config, "SCHEDULE_DEFAULT", "none")

    # Map schedule name to cv_utils scheduler
    if final_sched_name == "cosine":
        schedule = cv_utils.schedule_cosine
    elif final_sched_name == "exponential":
        schedule = cv_utils.schedule_exponential
    else:
        schedule = None

    w_final, final_loss = implementations.reg_logistic_regression(
        y_tr_01,
        X_tr,
        best_lambda,
        w0,
        max_iters=config.FINAL_MAX_ITERS,
        gamma=best_gamma,
        adam=final_use_adam,
        schedule=schedule,
        early_stopping=getattr(config, "EARLY_STOP_DEFAULT", False),
        patience=getattr(config, "PATIENCE_DEFAULT", 10),
        tol=getattr(config, "TOL_DEFAULT", 1e-6),
        verbose=False,
    )
    print(f"[Final] loss (unpenalized) = {final_loss:.6f}")

    np.save(config.SAVE_WEIGHTS, w_final)
    print(f"[Saved] Final weights -> {config.SAVE_WEIGHTS}")

    print(f"[Final Training] {time.time() - t_final:.1f}s")
    return w_final


def make_submission(X_te, w_final, best_thr, test_ids):
    """Generate predictions and write Kaggle-style submission CSV."""
    probs_te = implementations.sigmoid(X_te.dot(w_final))
    preds01_te = (probs_te >= best_thr).astype(int)
    preds_pm1_te = metrics.to_pm1_labels(preds01_te)
    helpers.create_csv_submission(test_ids, preds_pm1_te, config.OUTPUT_PRED)
    print(f"[Submission] saved -> {config.OUTPUT_PRED}")


 


def save_training_curves_with_holdout(X: np.ndarray, y01: np.ndarray,
                                      best_lambda: float, best_gamma: float,
                                      use_adam: bool | None, schedule_name: str | None) -> None:
    """Train on a train split, validate on a holdout; save loss curves to NPZ."""
    n = y01.shape[0]
    val_size = int(max(1, round(config.HOLDOUT_VAL_FRAC * n)))
    rng = np.random.RandomState(config.RNG_SEED)
    perm = rng.permutation(n)
    va_idx = perm[:val_size]
    tr_idx = perm[val_size:]
    X_tr_s, y_tr_s = X[tr_idx], y01[tr_idx]
    X_va_s, y_va_s = X[va_idx], y01[va_idx]

    # Resolve optimizer/schedule
    final_use_adam = use_adam if use_adam is not None else getattr(config, "USE_ADAM_DEFAULT", True)
    final_sched_name = schedule_name if schedule_name is not None else getattr(config, "SCHEDULE_DEFAULT", "none")
    if final_sched_name == "cosine":
        schedule = cv_utils.schedule_cosine
    elif final_sched_name == "exponential":
        schedule = cv_utils.schedule_exponential
    else:
        schedule = None

    w0 = np.zeros(X.shape[1], dtype=np.float32)
    losses_tr, losses_va = [], []

    def _cb(iter_idx, w, loss_tr, lr=None):
        losses_tr.append(float(loss_tr))
        # validate on holdout (unpenalized)
        losses_va.append(float(implementations.logistic_loss(y_va_s, X_va_s, w, lambda_=0)))

    # Train only on the train split (honest monitoring)
    implementations.reg_logistic_regression(
        y_tr_s,
        X_tr_s,
        best_lambda,
        w0,
        max_iters=config.FINAL_MAX_ITERS,
        gamma=best_gamma,
        adam=final_use_adam,
        schedule=schedule,
        early_stopping=getattr(config, "EARLY_STOP_DEFAULT", False),
        patience=getattr(config, "PATIENCE_DEFAULT", 10),
        tol=getattr(config, "TOL_DEFAULT", 1e-6),
        verbose=False,
        callback=_cb,
        val_data=(y_va_s, X_va_s) if getattr(config, "EARLY_STOP_DEFAULT", False) else None,
    )

    # Reconstruct LR curve if schedule is known
    if schedule is not None and len(losses_tr) > 0:
        lrs = np.array([schedule(best_gamma, t, config.FINAL_MAX_ITERS) for t in range(len(losses_tr))], dtype=np.float32)
    else:
        lrs = np.array([], dtype=np.float32)

    best_val_epoch = int(np.argmin(losses_va)) if len(losses_va) > 0 else -1
    curve_path = os.path.join(config.SAVE_DIR, "final_training_curve.npz")
    np.savez(
        curve_path,
        loss_train=np.array(losses_tr, dtype=np.float32),
        loss_val=np.array(losses_va, dtype=np.float32),
        lr=lrs,
        adam=final_use_adam,
        schedule=final_sched_name,
        lambda_=float(best_lambda),
        gamma=float(best_gamma),
        weighted_bce=bool(getattr(config, 'USE_WEIGHTED_BCE', False)),
        best_val_epoch=best_val_epoch,
        best_val_loss=(float(np.min(losses_va)) if len(losses_va) > 0 else np.nan),
        max_iters=int(config.FINAL_MAX_ITERS),
        seed=int(config.RNG_SEED),
        val_frac=float(config.HOLDOUT_VAL_FRAC),
        idx_val=va_idx,  # utile si tu veux recomposer des splits ensuite
    )
    print(f"[Saved] Training curves (train/val) -> {curve_path}")


def main() -> None:
    """Entrypoint: preprocess → tune → train → submit as toggled in config."""
    t = time.time()

    # Preprocessing: compute or load cached arrays
    if getattr(config, "PREPROCESSING", getattr(config, "DO_PREPROCESS", False)):
        x_train, x_test, y_train_pm1, train_ids, test_ids = helpers.load_csv_data(config.DATA_DIR)
        X_tr, X_te, y_tr_01 = preprocessing.preprocess2(
            x_train, x_test, y_train_pm1, train_ids, test_ids, config.PREPROC2_DATA_PATH
        )
    else:
        if not os.path.exists(config.PREPROC2_DATA_PATH):
            raise FileNotFoundError(f"{config.PREPROC2_DATA_PATH} not found.")
        npz = np.load(config.PREPROC2_DATA_PATH)
        X_tr = npz["X_train"]
        X_te = npz["X_test"]
        y_tr_01 = npz["y_train"]
        train_ids = npz["train_ids"]
        test_ids = npz["test_ids"]
        print(f"[Loaded] Preprocessed data from -> {config.PREPROC2_DATA_PATH}")
    print(f"[Preprocessing] {time.time() - t:.1f}s")

    # Hyperparameter tuning (or load best)
    if config.HYPERPARAM_TUNING:
        N_SPLITS = 5
        folds = cv_utils.stratified_kfold_indices(y_tr_01, n_splits=N_SPLITS, seed=config.RNG_SEED)
        best_lambda, best_gamma, best_thr, best_adam, best_sched = tune_hyperparameter(X_tr, y_tr_01, folds)
        print(f"[Tuning] lambda={best_lambda:.3e}, gamma={best_gamma:.3e}, thr={best_thr:.3f}, adam={best_adam}, schedule={best_sched}")
    else:
        if not os.path.exists(config.SAVE_BEST):
            raise FileNotFoundError(f"{config.SAVE_BEST} not found.")
        npz = np.load(config.SAVE_BEST, allow_pickle=False)
        best_lambda = float(npz["lambda_"])
        best_gamma = float(npz["gamma"])
        best_thr = float(npz["thr"])
        val_acc = float(npz["acc"])
        val_prec = float(npz["prec"])
        val_rec = float(npz["rec"])
        val_f1 = float(npz["f1"])
        best_adam = bool(npz["adam"]) if "adam" in npz.files else None
        best_sched = str(npz["schedule"]) if "schedule" in npz.files else None
        print(f"[Loaded] Best params from -> {config.SAVE_BEST}")
        print(
            f"[BEST] lambda={best_lambda:.3e}, gamma={best_gamma:.3e}, thr={best_thr:.3f}, "
            f"ACC={val_acc:.4f}, P={val_prec:.4f}, R={val_rec:.4f}, F1={val_f1:.4f}, "
            f"adam={best_adam}, schedule={best_sched}"
        )
        # Keep config.USE_WEIGHTED_BCE as the single source of truth; no override from saved params.
        print(f"[Config] USE_WEIGHTED_BCE={getattr(config, 'USE_WEIGHTED_BCE', False)}")

    # Final training + submission
    if config.DO_SUBMISSION:
        # Enregistre les courbes train/val avec un holdout (run séparé, honnête)
        if getattr(config, "HOLDOUT_VAL_FRAC", 0.0) > 0:
            save_training_curves_with_holdout(
                X_tr, y_tr_01, best_lambda, best_gamma, use_adam=best_adam, schedule_name=best_sched
            )
        # Entraînement final sur tout le train (pour la soumission)
        w_final = train_final_model(
            X_tr, y_tr_01, best_lambda, best_gamma, use_adam=best_adam, schedule_name=best_sched
        )
        make_submission(X_te, w_final, best_thr, test_ids)
        print(f"[TOTAL] {time.time() - t:.1f}s.")


if __name__ == "__main__":
    main()
