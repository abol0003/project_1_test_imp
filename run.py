import os
import time
import numpy as np
import multiprocessing as mp

import helpers
import implementations
import config
import metrics
import cv_utils


os.makedirs(config.SAVE_DIR, exist_ok=True)
np.random.seed(config.RNG_SEED)


def sample_loguniform(
    low: float, high: float, size: int, rng=np.random.RandomState(config.RNG_SEED)
) -> np.ndarray:
    """Sample values from a log-uniform distribution within a given range.

    Args:
        low (float): Lower bound of the sampling range.
        high (float): Upper bound of the sampling range.
        size (int): Number of samples to draw.
        rng (np.random.RandomState): Random number generator.

    Returns:
        np.ndarray: Array of sampled values.
    """
    lo, hi = np.log(low), np.log(high)
    return np.exp(rng.uniform(lo, hi, size))


def tune_hyperparameter(X_tr, y_tr_01, folds):
    """Perform hyperparameter tuning using cross-validation.

    This function searches for the best regularization and learning parameters
    by evaluating multiple configurations in parallel. Results are saved for
    reproducibility, and the best model is trained on the full dataset.

    Args:
        X_tr (np.ndarray): Training feature matrix.
        y_tr_01 (np.ndarray): Binary training labels (0/1).
        folds (list): Precomputed stratified K-fold indices.

    Returns:
        tuple: (best_lambda, best_gamma, best_thr, best_adam, best_sched, w_best)
    """
    t_tune = time.time()

    if config.NAGFREE_TUNING:
        lambda_samples = [3.96760508e-06]
        gamma_samples = np.full_like(lambda_samples, np.nan, dtype=float)
        adam_choices = [False]
        schedule_choices = ["none"]
    else:
        adam_choices = config.ADAM_CHOICES
        schedule_choices = config.SCHEDULE_CHOICES
        lambda_samples = (
            [3.96760508e-06]
            if schedule_choices is not None
            else sample_loguniform(
                config.LAMBDA_LOW, config.LAMBDA_HIGH, config.N_TRIALS
            )
        )
        gamma_samples = (
            [1]
            if schedule_choices is not None
            else sample_loguniform(config.GAMMA_LOW, config.GAMMA_HIGH, config.N_TRIALS)
        )

    print(
        "[TUNE] Using {} search".format(
            "NAG-Free" if config.NAGFREE_TUNING else "classic Adam/GD"
        )
    )

    tasks = [
        (
            y_tr_01,
            X_tr,
            folds,
            lam,
            gam,
            config.TUNING_MAX_ITERS,
            adam,
            sched,
            getattr(config, "EARLY_STOP_DEFAULT", False),
            getattr(config, "PATIENCE_DEFAULT", 10),
            getattr(config, "TOL_DEFAULT", 1e-6),
        )
        for lam, gam in zip(lambda_samples, gamma_samples)
        for adam in adam_choices
        for sched in schedule_choices
    ]

    nproc = max(1, (os.cpu_count() or 2) - 4)
    with mp.get_context("spawn").Pool(processes=nproc) as pool:
        results = pool.map(cv_utils.cv_train_and_eval, tasks)

    np.savez("tuning_results_backup.npz", results=results)

    best_pack = None
    best_score = -np.inf
    for idx, res in enumerate(results):
        lam, gam, thr, acc, prec, rec, f1 = res
        adam_choice = tasks[idx][6]
        sched_choice = tasks[idx][7]
        if f1 > best_score:
            best_score = f1
            best_pack = (lam, gam, thr, acc, prec, rec, f1, adam_choice, sched_choice)

    (
        best_lambda,
        best_gamma,
        best_thr,
        val_acc,
        val_prec,
        val_rec,
        val_f1,
        best_adam,
        best_sched,
    ) = best_pack

    if config.NAGFREE_TUNING:
        best_gamma = None
        best_adam = False
        best_sched = "none"

    print(
        f"[BEST-CV] λ={best_lambda:.3e}, γ={'—' if best_gamma is None else f'{best_gamma:.3e}'}, thr={best_thr:.3f}, "
        f"ACC={val_acc:.4f}, P={val_prec:.4f}, R={val_rec:.4f}, F1={val_f1:.4f}, "
        f"adam={best_adam}, schedule={best_sched}"
    )

    w_best = train_final_model(
        X_tr,
        y_tr_01,
        best_lambda,
        best_gamma,
        use_adam=best_adam,
        schedule_name=best_sched,
    )

    np.savez(
        config.SAVE_BEST,
        lambda_=best_lambda,
        gamma=np.nan if best_gamma is None else best_gamma,
        thr=best_thr,
        acc=float(val_acc) if val_acc is not None else np.nan,
        prec=float(val_prec) if val_prec is not None else np.nan,
        rec=float(val_rec) if val_rec is not None else np.nan,
        f1=float(val_f1),
        adam=best_adam,
        schedule=best_sched,
    )
    print(f"[Saved] Best params → {config.SAVE_BEST}")
    print(f"[Hyperparameter Tuning] done in {time.time() - t_tune:.1f}s")
    return best_lambda, best_gamma, best_thr, best_adam, best_sched, w_best


def train_kfold_and_predict_test(
    x_tr, y_tr, x_te, best_lambda, best_gamma=None, k_folds=5
):
    """Train using K-fold cross-validation and predict on the test set.

    Each fold trains a separate model, and test predictions are averaged across folds.
    The function identifies the best-performing fold and returns its model weights.

    Args:
        x_tr (np.ndarray): Training feature matrix.
        y_tr (np.ndarray): Binary training labels (0/1).
        x_te (np.ndarray): Test feature matrix.
        best_lambda (float): Regularization strength.
        best_gamma (float, optional): Learning rate (ignored for NAG-Free).
        k_folds (int): Number of cross-validation folds.

    Returns:
        tuple: (y_pred_test, test_probs, best_thr, w_list, w_best)
    """
    x_tr = np.asarray(x_tr)
    x_te = np.asarray(x_te)
    y_tr = np.asarray(y_tr).astype(int)

    folds = cv_utils.stratified_kfold_indices(
        y_tr, n_splits=k_folds, seed=config.RNG_SEED
    )
    oof_preds = np.zeros_like(y_tr, dtype=float)
    test_preds_all = []
    w_list = []

    use_nagfree = bool(getattr(config, "NAGFREE_TUNING", False))

    for k, (tr_idx, va_idx) in enumerate(folds):
        print(f"[K-Fold] Fold {k+1}/{k_folds}")
        xtr_k, ytr_k = x_tr[tr_idx], y_tr[tr_idx]
        xva_k, yva_k = x_tr[va_idx], y_tr[va_idx]

        w0 = np.zeros(xtr_k.shape[1], dtype=np.float32)

        if use_nagfree:
            w_k, _ = implementations.reg_logistic_regression_nagfree(
                ytr_k,
                xtr_k,
                lambda_=best_lambda,
                initial_w=w0,
                max_iters=config.FINAL_MAX_ITERS,
                tol=config.TOL_DEFAULT,
                L_max=config.NAGFREE_L_MAX,
                verbose=False,
            )
        else:
            w_k, _ = implementations.reg_logistic_regression(
                ytr_k,
                xtr_k,
                lambda_=best_lambda,
                initial_w=w0,
                max_iters=config.FINAL_MAX_ITERS,
                gamma=best_gamma if best_gamma is not None else 1,
                adam=config.USE_ADAM_DEFAULT,
                schedule=None,
                early_stopping=config.EARLY_STOP_DEFAULT,
                patience=config.PATIENCE_DEFAULT,
                tol=config.TOL_DEFAULT,
                verbose=False,
            )

        w_list.append(w_k)
        oof_preds[va_idx] = implementations.sigmoid(xva_k @ w_k)
        test_preds_all.append(implementations.sigmoid(x_te @ w_k))

    best_thr, _, _, best_f1 = cv_utils.best_threshold_by_f1(y_tr, oof_preds)
    print(f"[K-Fold] Optimal OOF threshold = {best_thr:.3f} | F1 = {best_f1:.4f}")

    test_probs = np.mean(test_preds_all, axis=0)
    y_pred_test = (test_probs >= best_thr).astype(int)

    fold_f1 = []
    for (tr_idx, va_idx), w_k in zip(folds, w_list):
        yva = y_tr[va_idx]
        pva = oof_preds[va_idx]
        yhat = (pva >= best_thr).astype(int)
        f1 = metrics.f1_score(yva, yhat)
        fold_f1.append(f1)
    best_fold = int(np.argmax(fold_f1))
    w_best = w_list[best_fold]
    print(f"[K-Fold] Best fold = {best_fold+1}/{k_folds} (F1={fold_f1[best_fold]:.4f})")

    return y_pred_test, test_probs, best_thr, w_list, w_best


def train_final_model(
    X_tr, y_tr_01, best_lambda, best_gamma, use_adam=None, schedule_name=None
):
    """Train the final logistic model on the entire training dataset.

    Depending on the configuration, the function trains either a standard logistic
    regression model or its NAG-Free variant, using the best hyperparameters found
    during tuning.

    Args:
        X_tr (np.ndarray): Training feature matrix.
        y_tr_01 (np.ndarray): Binary training labels (0/1).
        best_lambda (float): Regularization parameter.
        best_gamma (float): Learning rate.
        use_adam (bool, optional): Use Adam optimizer if True.
        schedule_name (str, optional): Learning rate schedule name.

    Returns:
        np.ndarray: Final trained model weights.
    """
    t_final = time.time()
    w0 = np.zeros(X_tr.shape[1], dtype=np.float32)

    if config.NAGFREE_TUNING:
        w_final, final_loss = implementations.reg_logistic_regression_nagfree(
            y_tr_01,
            X_tr,
            best_lambda,
            w0,
            max_iters=config.FINAL_MAX_ITERS,
            tol=config.TOL_DEFAULT,
            L_max=getattr(config, "NAGFREE_L_MAX", 1e8),
            verbose=False,
        )
        np.save(config.SAVE_WEIGHTS_NAGFREE, w_final)
        print(f"[Saved] Final weights → {config.SAVE_WEIGHTS_NAGFREE}")
    else:
        final_use_adam = use_adam if use_adam is not None else config.USE_ADAM_DEFAULT
        final_sched_name = (
            schedule_name if schedule_name is not None else config.SCHEDULE_DEFAULT
        )
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
            gamma=best_gamma if best_gamma is not None else 1e-2,
            adam=final_use_adam,
            schedule=schedule,
            early_stopping=config.EARLY_STOP_DEFAULT,
            patience=config.PATIENCE_DEFAULT,
            tol=config.TOL_DEFAULT,
            verbose=False,
        )
        np.save(config.SAVE_WEIGHTS, w_final)
        print(f"[Saved] Final weights → {config.SAVE_WEIGHTS}")

    print(f"[Final] loss (unpenalized) = {final_loss:.6f}")
    print(f"[Final Training] {time.time() - t_final:.1f}s")
    return w_final


def make_submission(X_te, w_final, best_thr, test_ids):
    """Generate and save predictions in CSV format.

    This function computes test predictions and converts them into Kaggle-style
    CSV submission files containing IDs and predicted labels.

    Args:
        X_te (np.ndarray): Test feature matrix.
        w_final (np.ndarray): Trained model weights.
        best_thr (float): Decision threshold.
        test_ids (np.ndarray): IDs corresponding to the test samples.
    """
    probs_te = implementations.sigmoid(X_te.dot(w_final))
    preds01_te = (probs_te >= best_thr).astype(int)
    preds_pm1_te = metrics.to_pm1_labels(preds01_te)
    helpers.create_csv_submission(test_ids, preds_pm1_te, config.OUTPUT_PRED)
    print(f"[Submission] saved → {config.OUTPUT_PRED}")


def main() -> None:
    """Main pipeline: preprocessing, hyperparameter tuning, final training, and submission."""
    t = time.time()

    if config.DO_PREPROCESS:
        import prep_main

        x_train, x_test, y_train_pm1, train_ids, test_ids = helpers.load_csv_data(
            config.DATA_DIR
        )
        X_tr, X_te, y_tr_01 = prep_main.preprocess2(
            x_train, x_test, y_train_pm1, train_ids, test_ids, config.PREPROC4_DATA_PATH
        )
    else:
        if not os.path.exists(config.PREPROC2_DATA_PATH):
            raise FileNotFoundError(f"{config.PREPROC2_DATA_PATH} not found.")
        npz = np.load(config.PREPROC2_DATA_PATH)
        X_tr, X_te, y_tr_01 = npz["X_train"], npz["X_test"], npz["y_train"]
        train_ids, test_ids = npz["train_ids"], npz["test_ids"]
        print(f"[Loaded] Preprocessed data from → {config.PREPROC4_DATA_PATH}")
    print(f"[Preprocessing] {time.time() - t:.1f}s")

    if config.HYPERPARAM_TUNING:
        N_SPLITS = 5
        folds = cv_utils.stratified_kfold_indices(
            y_tr_01, n_splits=N_SPLITS, seed=config.RNG_SEED
        )
        best_lambda, best_gamma, best_thr, best_adam, best_sched, w_best = (
            tune_hyperparameter(X_tr, y_tr_01, folds)
        )
        make_submission(X_te, w_best, best_thr, test_ids)
    else:
        npz = np.load(config.SAVE_BEST, allow_pickle=False)
        best_lambda = float(npz["lambda_"])
        best_gamma = float(npz["gamma"])
        best_thr = float(npz["thr"])
        best_adam = bool(npz["adam"]) if "adam" in npz.files else None
        best_sched = str(npz["schedule"]) if "schedule" in npz.files else None
        print(f"[Loaded] Best params from → {config.SAVE_WEIGHTS_NAGFREE}")

    if config.DO_SUBMISSION:
        if getattr(config, "CV_KFOLD_PREDICTION", False):
            print(
                "[K-Fold] Training final model with K-Fold CV and predicting test set"
            )
            y_pred_test, test_probs, best_thr, w_list, w_best = (
                train_kfold_and_predict_test(
                    X_tr, y_tr_01, X_te, best_lambda, best_gamma, k_folds=5
                )
            )
            make_submission(X_te, w_best, best_thr, test_ids)
        else:
            w_final = train_final_model(
                X_tr,
                y_tr_01,
                best_lambda,
                best_gamma,
                use_adam=best_adam,
                schedule_name=best_sched,
            )
            make_submission(X_te, w_final, best_thr, test_ids)

        print(f"[TOTAL] runtime {time.time() - t:.1f}s.")


if __name__ == "__main__":
    main()
