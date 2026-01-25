import numpy as np
import matplotlib.pyplot as plt

def projected_gradient_descent(y, A, alpha, c, x_init, lr=0.1, max_iters=500, tol=1e-6):
    """
    Solves x* = argmin ||y - h(Ax)||^2 subject to 0 <= x <= 1.
    """
    x = np.copy(x_init)
    loss_history = []
    
    for i in range(max_iters):
        # 1. Forward Pass
        z = np.dot(A, x)
        h_z = 1 / (1 + np.exp(-alpha * (z - c)))
        
        # 2. Compute Loss J(x)
        loss = np.sum((y - h_z)**2)
        loss_history.append(loss)
        
        # 3. Compute Gradient
        residual = y - h_z
        h_prime = alpha * h_z * (1 - h_z)
        grad = -2 * np.dot(A.T, (residual * h_prime))
        
        # 4. Gradient Step
        x_new = x - lr * grad
        
        # 5. Projection Step: Ensures 0 <= x <= 1
        x_new = np.clip(x_new, 0, 1)
        
        # 6. Convergence Check
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged at iteration {i}")
            break
            
        x = x_new
        
    return x, loss_history

# --- Simulation for "Experiments" Section ---
N = 50
A = np.random.randn(N, N)  # Replace with your specific convolution matrix A [cite: 17]
y = np.random.randint(0, 2, N) # Target binary vector y [cite: 23, 24]
alpha = 5.0
c = 0.5
x_start = np.random.rand(N)

x_opt, losses = projected_gradient_descent(y, A, alpha, c, x_start)

# Visualization for "Experiments" Section 
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.yscale('log')
plt.title("Convergence of Projected Gradient Descent")
plt.xlabel("Iterations")
plt.ylabel("Loss J(x)")
plt.grid(True)
plt.show()