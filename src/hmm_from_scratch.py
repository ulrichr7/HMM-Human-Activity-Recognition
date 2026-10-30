
import numpy as np

def logsumexp(a, axis=None, keepdims=False):
    a = np.asarray(a)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if keepdims:
        return out
    return np.squeeze(out, axis=axis)

def log_emission(X, means, variances):
    """Diagonal-covariance Gaussian log p(x_t | state)."""
    return -0.5 * (
        np.sum(np.log(2*np.pi*variances), axis=1)
        + ((X[:, None, :] - means[None, :, :])**2 / variances[None, :, :]).sum(axis=2)
    )

def forward_backward(X, pi, A, means, variances):
    """Log-space forward-backward; returns log-likelihood, gamma, summed xi."""
    T = len(X)
    K = len(pi)
    logA = np.log(np.clip(A, 1e-300, None))
    logpi = np.log(np.clip(pi, 1e-300, None))
    le = log_emission(X, means, variances)

    log_alpha = np.empty((T, K))
    log_beta = np.empty((T, K))
    log_alpha[0] = logpi + le[0]

    for t in range(1, T):
        log_alpha[t] = le[t] + logsumexp(
            log_alpha[t-1][:, None] + logA, axis=0
        )

    log_beta[-1] = 0.0
    for t in range(T-2, -1, -1):
        log_beta[t] = logsumexp(
            logA + le[t+1][None, :] + log_beta[t+1][None, :],
            axis=1
        )

    log_likelihood = float(logsumexp(log_alpha[-1]))
    gamma = np.exp(log_alpha + log_beta - log_likelihood)

    xi_sum = np.zeros((K, K))
    for t in range(T-1):
        log_xi = (
            log_alpha[t][:, None]
            + logA
            + le[t+1][None, :]
            + log_beta[t+1][None, :]
            - log_likelihood
        )
        xi_sum += np.exp(log_xi)

    return log_likelihood, gamma, xi_sum

def baum_welch(sequences, means, variances, A, pi,
               max_iter=100, tol=1e-5, min_variance=1e-3):
    """EM training with a log-likelihood convergence criterion."""
    K, D = means.shape
    previous = -np.inf
    history = []

    for iteration in range(max_iter):
        pi_acc = np.zeros(K)
        A_acc = np.zeros((K, K))
        gamma_sum = np.zeros(K)
        x_sum = np.zeros((K, D))
        x2_sum = np.zeros((K, D))
        total_ll = 0.0

        for _, X, _ in sequences:
            ll, gamma, xi_sum = forward_backward(
                X, pi, A, means, variances
            )
            total_ll += ll
            pi_acc += gamma[0]
            A_acc += xi_sum
            gamma_sum += gamma.sum(axis=0)
            x_sum += gamma.T @ X
            x2_sum += gamma.T @ (X**2)

        pi = pi_acc / pi_acc.sum()
        A = (A_acc + 1e-8) / (
            A_acc.sum(axis=1, keepdims=True) + K*1e-8
        )
        means = x_sum / gamma_sum[:, None]
        variances = np.maximum(
            x2_sum / gamma_sum[:, None] - means**2,
            min_variance
        )

        history.append(total_ll)

        if iteration > 0 and abs(total_ll - previous) < tol:
            return means, variances, A, pi, history

        previous = total_ll

    return means, variances, A, pi, history

def viterbi(X, pi, A, means, variances):
    """Decode the most likely hidden-state sequence."""
    T = len(X)
    K = len(pi)
    le = log_emission(X, means, variances)
    logpi = np.log(np.clip(pi, 1e-300, None))
    logA = np.log(np.clip(A, 1e-300, None))

    dp = np.empty((T, K))
    back = np.zeros((T, K), dtype=int)
    dp[0] = logpi + le[0]

    for t in range(1, T):
        scores = dp[t-1][:, None] + logA
        back[t] = np.argmax(scores, axis=0)
        dp[t] = le[t] + np.max(scores, axis=0)

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for t in range(T-2, -1, -1):
        path[t] = back[t+1, path[t+1]]

    return path, float(np.max(dp[-1]))
