import numpy as np

def h(z, alpha=10, c=0.5):
    """
    Implements the function h(z) = 1 / (1 + exp(-alpha * (z - c))).
    
    Parameters:
    - z: input value or numpy array (e.g., between 0 and 1).
    - alpha: stiffness parameter (alpha > 0).
    - c: cutoff threshold (c > 0).
    """
    return 1 / (1 + np.exp(-alpha * (z - c)))

def create_gaussian_kernel(size=7, sigma=1.5):
    """
    Creates a 1D Gaussian kernel theta normalized to sum to 1.
    """
    # Create coordinate range centered at 0
    x = np.linspace(-(size // 2), size // 2, size)
    # Gaussian formula: exp(-x^2 / (2 * sigma^2))
    kernel = np.exp(-0.5 * (x / sigma)**2)
    return kernel / kernel.sum()

def create_convolution_matrix_A(N, kernel):
    """
    Constructs an N x N convolution matrix A such that Ax = kernel * x.
    This creates a Toeplitz-like banded matrix.
    """
    A = np.zeros((N, N))
    k_len = len(kernel)
    pad = k_len // 2
    
    for i in range(N):
        for j in range(k_len):
            # Calculate the column index for the current kernel element
            col_idx = i - pad + j
            # Apply zero-padding boundary conditions
            if 0 <= col_idx < N:
                A[i, col_idx] = kernel[j]
    return A

# --- Example Usage ---

# 1. Setup Parameters
N = 64
alpha_val = 20
c_val = 0.5


# Initialize y with zeros
y = np.zeros(N, dtype=int)

# Set indices 8 through 16 to 1
y[8:17] = 1

# Set indices 32 through 48 to 1
y[32:49] = 1

def f_theta(x, A, alpha=20, c=0.5):
    """
    Implements the function f_theta(x) = h(Ax).
    
    Parameters:
    - x: Input signal/vector (shape: (N,)).
    - A: The convolution matrix (shape: (N, N)).
    - alpha: The stiffness parameter for the h(z) function.
    - c: The threshold cutoff for the h(z) function.
    """
    # Step 1: Compute z = Ax (The convolution/blurring step)
    z = np.dot(A, x)
    
    # Step 2: Apply the non-linear thresholding function h(z)
    # Using our previously defined h function logic:
    return 1 / (1 + np.exp(-alpha * (z - c)))

# --- Full Pipeline Example ---
# 1. Define input y (the binary pulses)
y = np.zeros(64)
y[8:17] = 1
y[32:49] = 1

# 2. Get matrix A (assuming it was constructed with Gaussian kernel theta)
# output = f_theta(y, A_matrix, alpha=20, c=0.5)
# 2. Evaluate h(z)
z_test = np.linspace(0, 1, 5)
h_test = h(z_test, alpha=alpha_val, c=c_val)

# 3. Construct Matrix A
theta = create_gaussian_kernel(size=7, sigma=1.5)
A = create_convolution_matrix_A(N, theta)

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Support Functions ---
def h(z, alpha=20, c=0.5):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def create_gaussian_kernel(size=7, sigma=1.5):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    return kernel / kernel.sum()

def create_A(N, kernel):
    A = np.zeros((N, N))
    k_len = len(kernel)
    pad = k_len // 2
    for i in range(N):
        for j in range(k_len):
            col_idx = i - pad + j
            if 0 <= col_idx < N:
                A[i, col_idx] = kernel[j]
    return A

# --- 2. Setup Data ---
N = 64
y = np.zeros(N)
y[8:17] = 1   # Indices 8 to 16
y[32:49] = 1  # Indices 32 to 48


def J(x, y, A, alpha=20, c=0.5):
    """
    Implements the cost function J(x) = ||y - f_theta(x)||_2^2.
    
    Parameters:
    - x: The input vector we are evaluating (the 'candidate' signal).
    - y: The target binary vector (the 'ground truth').
    - A: The N x N convolution matrix.
    - alpha: Stiffness parameter for the activation function h.
    - c: Cutoff threshold for the activation function h.
    """
    # 1. Compute f_theta(x) = h(Ax)
    z = np.dot(A, x)
    f_x = 1 / (1 + np.exp(-alpha * (z - c)))
    
    # 2. Compute the residual (difference)
    residual = y - f_x
    
    # 3. Return the squared L2 norm (sum of squares)
    return np.sum(residual**2)

# --- Example Usage ---
# Assuming y and A are already defined from previous steps:
# current_cost = J(x_candidate, y, A_matrix, alpha=20, c=0.5)

# Create matrix A and compute f(y)
f_y = h(np.dot(A, y), alpha=20, c=0.5)

# # --- 3. Plotting ---
# plt.figure(figsize=(12, 6))

# # Use drawstyle='steps-mid' to show the discrete binary nature of y
# plt.plot(range(N), y, label='Original $y$ (Binary)', color='blue', 
#          linewidth=2, linestyle='--', drawstyle='steps-mid', alpha=0.6)

# # Plot f(y) as a solid line
# plt.plot(range(N), f_y, label='$f_{\\theta}(y) = h(Ay)$', color='red', 
#          linewidth=2.5)

# plt.title('Comparison: Binary Vector $y$ vs. Transformed $f_{\\theta}(y)$')
# plt.xlabel('Vector Index')
# plt.ylabel('Value')
# plt.legend(loc='upper right')
# plt.grid(True, linestyle=':', alpha=0.7)
# plt.ylim(-0.1, 1.1)

# plt.show()
def J(x_vec, y, A, alpha=20, c=0.5):
    f_x = h(np.dot(A, x_vec), alpha, c)
    return np.sum((y - f_x)**2)

# --- 3. Varying x (as a scalar multiplier of a vector of ones) ---
gamma_range = np.linspace(0, 1.2, 100)
costs = [J(np.ones(N) * g, y, A) for g in gamma_range]

# --- 4. Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(gamma_range, costs, color='purple', linewidth=2, label='$J(\gamma \cdot \mathbf{1})$')

# Add marker for the cutoff c
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.6, label='Cutoff $c=0.5$')

plt.title(r'Cost Function $J(x)$ Profile')
plt.xlabel(r'Scalar Input Magnitude ($\gamma$)')
plt.ylabel(r'Cost Value $J(x)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()