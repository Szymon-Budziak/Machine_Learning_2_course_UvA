

def M_step(self, X, gamma, xi):
    """
    Performs the M-step to update the parameters based on gamma and xi.
    """
    T, k = gamma.shape

    # Update initial state probabilities
    self.pi = gamma[0]

    # Update transition matrix
    self.A = np.sum(xi, axis=0) / np.sum(xi, axis=(0, 1))[:, None]

    # Update emission parameters (mean and covariance)
    self.mu = np.sum(gamma[:, :, None] * X[:, None, :], axis=0) / np.sum(gamma, axis=0)[:, None]
    self.S = np.zeros((k, X.shape[1], X.shape[1]))
    for i in range(k):
        diff = X - self.mu[i]
        self.S[i] = (gamma[:, i][:, None, None] * (diff[:, :, None] @ diff[:, None, :])).sum(axis=0)
        self.S[i] /= gamma[:, i].sum()

def M_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
    """
    Performs the M-step to update the parameters of the HMM.

    Args:
        X (np.ndarray): The observed data. Shape (T, d).
        gamma (np.ndarray): The responsibilities. Shape (T, k).
        xi (np.ndarray): The transition values. Shape (T-1, k, k).
    """
    # Update transition matrix A
    gamma_sum = np.sum(gamma, axis=0)  # Sum of responsibilities for each state
    self.mu = (gamma.T @ X) / gamma_sum[:, None]  # Weighted average of observations

    # Update covariances Sigma
    self.S = np.zeros((k, d, d))  # Initialize covariance matrices
    for i in range(k):
        X_centered = X - self.mu[i]  # Centered data
        self.S[i] = (gamma[:, i][:, None] * X_centered).T @ X_centered / gamma_sum[i]  # Weighted covariance
