import numpy as np
import config
from helpers import *

# Tracks if early stopping fired, at which iteration, the best monitored loss, and the monitor source.
EARLY_STOP_META = {"triggered": False, "iter": None, "best_loss": None, "monitor": None}


# =========================================================
# 1) SIX CORE FUNCTIONS 
# =========================================================

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Batch gradient descent for MSE minimization.

    Iteratively updates parameters using full-batch gradients of the quadratic loss.
    Serves as a baseline optimizer for least-squares problems.

    Args:
        y: array-like, shape (N,). Targets.
        tx: array-like, shape (N, D). Design matrix.
        initial_w: array-like, shape (D,). Initial parameters.
        max_iters: int. Number of iterations.
        gamma: float. Learning rate.

    Returns:
        tuple: (w, loss)
            w: array-like, shape (D,). Final parameters.
            loss: float. Final MSE on (y, tx, w).
    """
    ws = [initial_w]
    w = initial_w
    loss = compute_loss(y, tx, w)

    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w -= gamma * grad
        # keep history of parameters
        ws.append(w)
        # losses.append(loss)
    loss = compute_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic gradient descent for MSE minimization.

    Uses a single randomly sampled instance per step to estimate the gradient,
    which can speed up convergence on large datasets.

    Args:
        y: array-like, shape (N,). Targets.
        tx: array-like, shape (N, D). Design matrix.
        initial_w: array-like, shape (D,). Initial parameters.
        max_iters: int. Number of iterations.
        gamma: float. Learning rate.

    Returns:
        tuple: (w, loss)
            w: array-like, shape (D,). Final parameters.
            loss: float. Final MSE on (y, tx, w).
    """
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
    """
    Ordinary least squares via normal equations.

    Solves (X^T X) w = X^T y and returns the MSE on the fitted weights.

    Args:
        y: array-like, shape (N,). Targets.
        tx: array-like, shape (N, D). Design matrix.

    Returns:
        tuple: (w, loss)
            w: array-like, shape (D,). Closed-form solution.
            loss: float. MSE on (y, tx, w).
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)  # w = (X^T X)^{-1} X^T y
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression (L2-regularized least squares) via normal equations.

    Solves (X^T X + 2 N λ I) w = X^T y to improve conditioning and reduce variance.

    Args:
        y: array-like, shape (N,). Targets.
        tx: array-like, shape (N, D). Design matrix.
        lambda_: float. L2 regularization strength.

    Returns:
        tuple: (w, loss)
            w: array-like, shape (D,). Regularized solution.
            loss: float. MSE on (y, tx, w).
    """
    N, D = tx.shape
    A = tx.T.dot(tx) + 2 * N * lambda_ * np.identity(D)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform unregularized logistic regression using gradient descent.

    This function minimizes the binary cross-entropy loss to estimate the model weights.
    It optionally handles class imbalance automatically through weighted gradients,
    defined inside logistic_gradient.

    Args:
        y (np.ndarray): Binary target vector of shape (N,).
        tx (np.ndarray): Design matrix of shape (N, D).
        initial_w (np.ndarray): Initial weight vector of shape (D,).
        max_iters (int): Maximum number of iterations.
        gamma (float): Learning rate.

    Returns:
        tuple:
            - w (np.ndarray): Optimized weight vector of shape (D,).
            - loss (float): Final unpenalized logistic loss.
    """
    w = initial_w
    for _ in range(max_iters):
        grad = logistic_gradient(y, tx, w, lambda_=0)
        w -= gamma * grad
    return w, logistic_loss(y, tx, w)


def reg_logistic_regression(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    gamma,
    adam=False,
    schedule=None,
    early_stopping=False,
    patience=10,
    tol=1e-3,
    verbose=False,
    callback=None,
    val_data=None,
):
    """Perform L2-regularized logistic regression with optional Adam optimization and early stopping.

    This variant extends standard logistic regression with an L2 penalty to prevent overfitting.
    The function supports Adam optimization, learning-rate scheduling, and early stopping based on
    validation loss. Class imbalance adjustments are handled internally through logistic_gradient.

    Args:
        y (np.ndarray): Binary target vector of shape (N,).
        tx (np.ndarray): Design matrix of shape (N, D).
        lambda_ (float): L2 regularization coefficient.
        initial_w (np.ndarray): Initial weight vector of shape (D,).
        max_iters (int): Maximum number of training iterations.
        gamma (float): Base learning rate.
        adam (bool): If True, use Adam optimizer; otherwise, standard gradient descent.
        schedule (callable, optional): Learning rate scheduler function.
        early_stopping (bool): If True, stop training when validation loss stops improving.
        patience (int): Number of iterations to wait before stopping.
        tol (float): Convergence tolerance.
        verbose (bool): If True, print training progress.
        callback (callable, optional): Optional function called at each iteration.
        val_data (tuple, optional): Optional (y_val, X_val) for validation monitoring.

    Returns:
        tuple:
            - w (np.ndarray): Final optimized weights.
            - loss (float): Final unpenalized training loss.
    """
    w = initial_w.astype(np.float32, copy=True)
    m, v = np.zeros_like(w), np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_loss, best_w, wait, best_iter = np.inf, w.copy(), 0, 0

    if verbose:
        print(
            f"[Train] adam={adam} schedule={'on' if schedule else 'none'} early_stop={early_stopping} "
            f"lambda={lambda_:.3e} gamma={gamma:.3e} iters={max_iters}"
        )

    if val_data is not None:
        y_val, X_val = np.asarray(val_data[0]), np.asarray(val_data[1])
    else:
        y_val, X_val = None, None

    monitor_kind = "val" if y_val is not None else "train"
    global EARLY_STOP_META
    EARLY_STOP_META = {"triggered": False, "iter": None, "best_loss": None, "monitor": monitor_kind}

    for t in range(1, max_iters + 1):
        lr = schedule(gamma, t - 1, max_iters) if schedule else gamma
        grad = logistic_gradient(y, tx, w, lambda_=0)
        g_reg = grad + 2.0 * lambda_ * w

        if adam:
            m = b1 * m + (1 - b1) * g_reg
            v = b2 * v + (1 - b2) * (g_reg * g_reg)
            m_hat = m / (1 - b1**t)
            v_hat = v / (1 - b2**t)
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            w -= lr * g_reg

        cur_train_loss = logistic_loss(y, tx, w, lambda_=0)
        cur_monitor_loss = (
            logistic_loss(y_val, X_val, w, lambda_=0) if y_val is not None else cur_train_loss
        )

        if callback:
            try:
                callback(t, w, float(cur_train_loss), float(lr))
            except Exception:
                pass

        if early_stopping and t > 150:
            if cur_monitor_loss + 1e-2 < best_loss:
                best_loss, best_w, best_iter, wait = cur_monitor_loss, w.copy(), t, 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(
                            f"[EarlyStop] iter={t} best_monitor={best_loss:.6f}"
                            f"{' (val)' if y_val is not None else ' (train)'}"
                        )
                    w = best_w
                    EARLY_STOP_META.update(
                        {
                            "triggered": True,
                            "iter": int(best_iter),
                            "best_loss": float(best_loss),
                            "monitor": monitor_kind,
                        }
                    )
                    break

        if verbose and (t % max(1, max_iters // 10) == 0):
            pen = logistic_loss(y, tx, w, lambda_=lambda_)
            print(
                f"[Iter {t}/{max_iters}] train_unpen={cur_train_loss:.6f} "
                f"monitor={'val' if y_val is not None else 'train'}={cur_monitor_loss:.6f} "
                f"pen={pen:.6f}"
            )

    final_loss = logistic_loss(y, tx, w, lambda_=0)
    if early_stopping and EARLY_STOP_META["iter"] is None:
        last_monitor = logistic_loss(y_val, X_val, w, lambda_=0) if y_val is not None else final_loss
        EARLY_STOP_META.update(
            {
                "triggered": False,
                "iter": int(best_iter if best_iter > 0 else t),
                "best_loss": float(best_loss if np.isfinite(best_loss) else last_monitor),
                "monitor": monitor_kind,
            }
        )
    return w, final_loss



# =========================================================
# 2) ADDITIONAL FUNCTIONS
# =========================================================

def sigmoid(z):
    """
    sigmoid.

    Clips inputs to limit overflow in exp.

    Args:
        z: scalar or array-like. Input.

    Returns:
        scalar or np.ndarray: Sigmoid activation.
    """
    # clip for numerical stability
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(y, tx, w, lambda_=0):
    """
    Binary cross-entropy with optional L2 penalty.

    Computes the negative log-likelihood of the logistic model. When lambda_ > 0,
    adds λ ||w||^2 to the objective (penalty applied to all weights).

    Args:
        y: array-like in {0,1}, shape (N,). Binary labels.
        tx: array-like, shape (N, D). Design matrix.
        w: array-like, shape (D,). Parameters.
        lambda_: float. L2 penalty strength.

    Returns:
        float: Logistic loss value.
    """
    sig = sigmoid(tx.dot(w))
    eps = 1e-12
    loss = -np.mean(y * np.log(sig + eps) + (1 - y) * np.log(1 - sig + eps))
    if lambda_ > 0:
        loss += lambda_ * np.sum(w**2)
    return loss


def logistic_gradient(y, tx, w, lambda_=0):
    """Compute the gradient of the logistic loss with optional L2 penalty and class weighting.

    This function computes the gradient of the binary cross-entropy loss used in logistic regression.
    If config.USE_WEIGHTED_BCE is enabled, it applies class-balanced weights to handle label imbalance.
    When lambda_ is greater than zero, an L2 regularization term is added to the gradient to reduce
    overfitting.

    Args:
        y (np.ndarray): Binary target vector of shape (N,).
        tx (np.ndarray): Input feature matrix of shape (N, D).
        w (np.ndarray): Current weight vector of shape (D,).
        lambda_ (float, optional): Regularization coefficient (default is 0).

    Returns:
        np.ndarray: Gradient vector of shape (D,).
    """
    y = np.asarray(y)
    p = sigmoid(tx.dot(w))
    resid = p - y

    if getattr(config, "USE_WEIGHTED_BCE", False):
        n_tot = float(y.size)
        n_pos = float(np.sum(y))
        n_neg = n_tot - n_pos
        a_pos = n_tot / (2.0 * max(1.0, n_pos))
        a_neg = n_tot / (2.0 * max(1.0, n_neg))
        w_samp = (y * a_pos + (1.0 - y) * a_neg).astype(np.float32, copy=False)
        denom_w = float(np.sum(w_samp))
        grad = tx.T.dot(resid * w_samp) / denom_w
    else:
        grad = tx.T.dot(resid) / y.size

    if lambda_ > 0:
        grad += 2 * lambda_ * w
    return grad



def compute_loss(y, tx, w):
    """
    Mean squared error (MSE) objective for linear regression.

    Returns 0.5 * mean((y - Xw)^2), a standard quadratic objective useful for
    baseline comparisons and closed-form solvers.

    Args:
        y: array-like, shape (N,). Targets.
        tx: array-like, shape (N, D). Design matrix.
        w: array-like, shape (D,). Parameters.

    Returns:
        float: MSE value.
    """
    err = y - tx.dot(w)
    return 0.5 * np.mean(err**2)  # np.mean(np.abs(err)) for MAE


def compute_gradient(y, tx, w, sgd=False):
    """
    Gradient of the MSE objective.

    Computes the batch gradient X^T (Xw − y) / N, or the single-sample version
    when sgd=True (no averaging). Used by GD/SGD for least squares.

    Args:
        y: array-like or scalar. Targets (vector for batch, scalar for SGD).
        tx: array-like. Design matrix or single row for SGD.
        w: array-like, shape (D,). Parameters.
        sgd: bool. Use single-sample gradient if True.

    Returns:
        np.ndarray: Gradient array of shape (D,).
    """
    err = y - tx.dot(w)
    if sgd:
        grad = -(tx.T.dot(err))
    else:
        grad = -(tx.T.dot(err)) / len(y)
    return grad


# =========================================================
# 3) NAG-FREE SECTION
# =========================================================

def reg_logistic_regression_nagfree(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    tol=1e-8,
    L_max=None,
    verbose=False,
    callback=None,
    val_data=None,
):
    """
    L2-regularized logistic regression trained with the NAG-Free optimizer.

    Delegates step-size and curvature handling to NAG-Free, which adaptively tracks
    local smoothness to accelerate convergence without explicit momentum tuning.
    The unpenalized data loss is returned for consistency with downstream thresholding.

    Args:
        y: array-like in {0,1}, shape (N,). Binary labels.
        tx: array-like, shape (N, D). Design matrix.
        lambda_: float. L2 regularization strength.
        initial_w: array-like, shape (D,). Initial parameters.
        max_iters: int. Maximum iterations.
        tol: float. Stopping tolerance used by NAG-Free.
        L_max: float or None. Upper bound for local Lipschitz estimate.
        verbose: bool. Verbose output inside NAG-Free (if supported).
        callback: callable or None. Optional hook; ignored here but kept for API parity.
        val_data: tuple or None. Optional (y_val, X_val) for compatibility.

    Returns:
        tuple: (w, loss)
            w: array-like, shape (D,). Final parameters from NAG-Free.
            loss: float. Unpenalized logistic loss on (y, tx, w).
    """
    from extensions import nagfree  

    w0 = np.asarray(initial_w, dtype=np.float32, copy=True)
    print(f"y size : {y.size}, tx shape: {tx.shape}, w0 shape: {w0.shape}")
    # L2-penalized gradient (same convention as reg_logistic_regression)
    def grad_fn(w):
        p = sigmoid(tx.dot(w))
        grad = tx.T.dot(p - y) / y.size
        grad += 2.0 * lambda_ * w
        return grad


    # NAG-Free call (handles its own step-size and curvature)
    w_final = nagfree(
        x0=w0,
        g=grad_fn,
        maxit=int(max_iters),
        tol=float(tol),
        L_max=float(L_max if L_max is not None else getattr(config, "NAGFREE_L_MAX", 1e8)),
        track_history=False,
    )

    final_loss = logistic_loss(y, tx, w_final, lambda_=0)
    return w_final, final_loss
