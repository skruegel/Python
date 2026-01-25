import numpy as np

def compute_gradient(x, y, A, alpha, c):
    """
    Computes the gradient of J(x) = ||y - h(Ax)||^2 with respect to x.
    
    Parameters:
    x (ndarray): Current input estimate (N,)
    y (ndarray): Target binary vector (N,)
    A (ndarray): Convolution matrix (N, N)
    alpha (float): Stiffness parameter
    c (float): Cutoff threshold
    
    Returns:
    grad (ndarray): Gradient vector (N,)
    """
    # 1. Forward Pass: Compute z = Ax
    z = np.dot(A, x) [cite: 15, 17]
    
    # 2. Activation: Compute h(z)
    # h(z) = 1 / (1 + exp(-alpha * (z - c)))
    h_z = 1 / (1 + np.exp(-alpha * (z - c))) [cite: 19]
    
    # 3. Compute the residual: (y - f_theta(x))
    residual = y - h_z [cite: 23, 24]
    
    # 4. Derivative of the sigmoid: h'(z) = alpha * h(z) * (1 - h(z))
    # This represents the "stiffness" impact on the gradient
    h_prime = alpha * h_z * (1 - h_z) [cite: 20]
    
    # 5. Backward Pass: Chain Rule
    # Combining the outer gradient, the activation derivative, and A transpose
    # grad = -2 * A.T @ (residual * h_prime)
    grad = -2 * np.dot(A.T, (residual * h_prime)) [cite: 29]
    
    return grad