import numpy as np
import config
from helpers import *

# Last training metadata (read-only for callers)
EARLY_STOP_META = {"triggered": False, "iter": None, "best_loss": None, "monitor": None}


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    "hey"
    ws = [initial_w]
    w = initial_w
    loss = compute_loss(y, tx, w)

    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w -= gamma * grad
        # store w and loss
        ws.append(w)
        # losses.append(loss)
    loss = compute_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    "hey"
    w = initial_w
    ws = [w]
    loss = compute_loss(y, tx, w)

    for _ in range(max_iters):
        idx = np.random.randint(0, len(y))
        y_b = y[idx]
        tx_b = tx[idx]

        grad = compute_gradient(y_b, tx_b, w, sgd=True)
        w -= gamma * grad

        # loss = compute_loss(y, tx, w)
        # losses.append(loss)
        ws.append(w)
    loss = compute_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    "hey"
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)  # w=a^-1 x b
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    "hey"
    N, D = tx.shape
    A = tx.T.dot(tx) + 2 * N * lambda_ * np.identity(D)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    "hey"
    w = initial_w
    use_weighted = getattr(config, "USE_WEIGHTED_BCE", False)
    if use_weighted:
        # class-balanced weights for y in {0,1}
        n_pos = float(np.sum(y))
        n_tot = float(y.size)
        n_neg = n_tot - n_pos
        a_pos = n_tot / (2.0 * max(1.0, n_pos))
        a_neg = n_tot / (2.0 * max(1.0, n_neg))
        w_samp = (y * a_pos + (1.0 - y) * a_neg).astype(np.float32, copy=False)
        denom_w = float(np.sum(w_samp))
        for _ in range(max_iters):
            p = sigmoid(tx.dot(w))
            resid = (p - y)
            grad = tx.T.dot(resid * w_samp) / denom_w
            w -= gamma * grad
    else:
        for _ in range(max_iters):
            w -= gamma * logistic_gradient(y, tx, w)
    return w, logistic_loss(y, tx, w)


# def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
#     "hey"
#     w = initial_w
#     for _ in range(max_iters):
#         w -= gamma * logistic_gradient(y, tx, w, lambda_=lambda_)
#     return w, logistic_loss(y, tx, w, lambda_=0)

def reg_logistic_regression(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    gamma,
    adam=True,
    schedule=None,
    early_stopping=False,
    patience=10,
    tol=1e-3,
    verbose=False,
    callback=None,
    val_data=None,
):
    """Regularized logistic regression (L2) with options for Adam, LR schedule, and early stopping.

    Extras (backward-compatible):
    - If config.USE_WEIGHTED_BCE is True, use class-balanced weights in the BCE gradient
      (alpha_pos = N/(2*N_pos), alpha_neg = N/(2*N_neg)). Penalty stays as lambda * sum(w**2).
    - Early stopping se base sur la validation loss si val_data est fourni.
    """
    w = initial_w.astype(np.float32, copy=True)

    # Adam buffers (used if adam=True)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8

    best_loss = np.inf
    best_w = w.copy()
    wait = 0

    n = y.size
    if (verbose):
        print(
            f"[Train] adam={adam} schedule={'on' if schedule else 'none'} early_stop={early_stopping} "
            f"lambda={lambda_:.3e} gamma={gamma:.3e} iters={max_iters}"
        )

    # Optional class-balanced weighting (no API change: toggled via config)
    use_weighted = config.USE_WEIGHTED_BCE
    if use_weighted:
        # y expected in {0,1}
        n_pos = float(np.sum(y))
        n_tot = float(y.size)
        n_neg = n_tot - n_pos
        # avoid division by zero in extreme edge-cases
        a_pos = n_tot / (2.0 * max(1.0, n_pos))
        a_neg = n_tot / (2.0 * max(1.0, n_neg))
        w_samp = (y * a_pos + (1.0 - y) * a_neg).astype(np.float32, copy=False)
        denom_w = float(np.sum(w_samp))
    else:
        w_samp = None
        denom_w = float(y.size)

    # Unpack optional validation data if provided
    if val_data is not None:
        y_val, X_val = val_data
        y_val = np.asarray(y_val)
        X_val = np.asarray(X_val)
    else:
        y_val, X_val = None, None

    # Monitoring mode for early stopping: prefer val if provided, else train
    monitor_kind = "val" if (y_val is not None) else "train"
    best_iter = 0
    if early_stopping and (y_val is None or X_val is None):
        if verbose:
            print("[EarlyStop] val_data not provided; falling back to training loss as monitor.")

    # Reset global metadata
    global EARLY_STOP_META
    EARLY_STOP_META = {"triggered": False, "iter": None, "best_loss": None, "monitor": monitor_kind}

    for t in range(1, max_iters + 1):
        lr = schedule(gamma, t - 1, max_iters) if schedule else gamma
        y_b = y
        tx_b = tx

        # plain logistic gradient
        p = sigmoid(tx_b.dot(w))
        resid = (p - y_b)
        if use_weighted:
            # Weighted BCE gradient: X^T((p - y) * w_i) / sum(w_i)
            grad = tx_b.T.dot(resid * w_samp) / denom_w
        else:
            grad = tx_b.T.dot(resid) / y_b.size
        # add L2 penalty (on all weights; bias included)
        g_reg = grad.copy()
        g_reg += 2.0 * lambda_ * w

        if adam:
            m = b1 * m + (1 - b1) * g_reg
            v = b2 * v + (1 - b2) * (g_reg * g_reg)
            m_hat = m / (1 - b1**t)
            v_hat = v / (1 - b2**t)
            w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            w = w - lr * g_reg

        # Compute training loss for logging/callback
        cur_train_loss = logistic_loss(y, tx, w, lambda_=0)
        # Monitor strictly the validation loss when early_stopping is enabled
        cur_monitor_loss = (
            logistic_loss(y_val, X_val, w, lambda_=0) if (y_val is not None) else cur_train_loss
        )
        if callback is not None:
            try:
                callback(t, w, float(cur_train_loss), float(lr))
            except Exception:
                pass

        if early_stopping and t>49:  # wait at least 50 iters
            if cur_monitor_loss + tol < best_loss:
                best_loss = cur_monitor_loss
                best_w = w.copy()
                best_iter = t
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"[EarlyStop] iter={t} best_monitor={best_loss:.6f}{' (val)' if y_val is not None else ' (train)'}")
                    w = best_w
                    # persist early-stop info
                    EARLY_STOP_META.update({
                        "triggered": True,
                        "iter": int(best_iter),
                        "best_loss": float(best_loss),
                        "monitor": monitor_kind,
                    })
                    break

        if (verbose) and (t % max(1, max_iters // 10) == 0):
            pen = logistic_loss(y, tx, w, lambda_=lambda_)
            print(
                f"[Iter {t}/{max_iters}] train_unpen={cur_train_loss:.6f} monitor={'val' if y_val is not None else 'train'}"
                f"={cur_monitor_loss:.6f} pen={pen:.6f}"
            )

    final_loss = logistic_loss(y, tx, w, lambda_=0)
    # If no break occurred, still record the best seen (or last) iteration
    if early_stopping and EARLY_STOP_META["iter"] is None:
        last_monitor = logistic_loss(y_val, X_val, w, lambda_=0) if y_val is not None else final_loss
        EARLY_STOP_META.update({
            "triggered": False,
            "iter": int(best_iter if best_iter > 0 else t),
            "best_loss": float(best_loss if np.isfinite(best_loss) else last_monitor),
            "monitor": monitor_kind,
        })
    return w, final_loss



## Additional function computed#####


def sigmoid(z):
    # clip for numerical stability
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(y, tx, w, lambda_=0):
    sig = sigmoid(tx.dot(w))
    eps = 1e-12
    loss = -np.mean(y * np.log(sig + eps) + (1 - y) * np.log(1 - sig + eps))
    if lambda_ > 0:
        loss += lambda_ * np.sum(w**2)
    return loss


def logistic_gradient(y, tx, w, lambda_=0):
    """Plain logistic gradient; optional L2 on all weights when lambda_>0 (kept for backward compat).
    Note: reg_logistic_regression handles L2 without penalizing bias; callers relying on this helper
    should be aware this version penalizes all weights when lambda_>0.
    """
    grad = tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)
    if lambda_ > 0:
        grad += 2 * lambda_ * w
    return grad


## Additional function needed taken from lab 2


def compute_loss(y, tx, w):
    err = y - tx.dot(w)
    return 0.5 * np.mean(err**2)  # np.mean(np.abs(err)) for MAE


def compute_gradient(y, tx, w, sgd=False):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    if sgd:
        grad = -(tx.T.dot(err))
    else:
        grad = -(tx.T.dot(err)) / len(y)
    return grad
