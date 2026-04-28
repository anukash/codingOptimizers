"""
Gradient Descent Optimizer Implementation.

This module provides implementations of gradient descent algorithms for
optimization. It includes support for various variants and learning rate
scheduling strategies.

Author: Anurag Kashyap
Created: 26-Apr-2026
"""

# Mathematical Formulation
# ========================
# Standard Gradient Descent Update Rule:
#   θ(t+1) = θ(t) - η·∇f(θ(t))
#
# Where:
#   θ(t)     : Parameters at iteration t
#   η        : Learning rate
#   ∇f(θ(t)) : Gradient of objective function f at θ(t)

import numpy as np

def gradient_descent(loss_fn, grad_fn, theta_init, lr = 0.01, steps = 1000):
    """
    Perform gradient descent optimization.

    Parameters:
    - loss_fn    : Function to compute the loss given parameters.
    - grad_fn    : Function to compute the gradient of the loss with respect to parameters.
    - theta_init : Initial parameters (numpy array).
    - lr         : Learning rate (float).
    - steps      : Number of iterations (int).

    Returns:
    - theta      : Optimized parameters after gradient descent.
    """
    theta   = theta_init.copy()
    history = []
    
    for step in range(steps):
        grad   = grad_fn(theta)   # Compute the gradient
        theta -= lr * grad        # Update parameters
        
        # Optionally, print loss every 10 steps
        if step % 10 == 0:
            loss = loss_fn(theta).item()  # Extract scalar value
            history.append(loss)
            print(f"Step {step}, Loss: {loss:.4f}")
    
    return theta, history


if __name__ == "__main__":
    # ================================================================
    # EXAMPLE 1: Multi-parameter optimization (2D)
    # ================================================================
    print("\n" + "="*60)
    print("EXAMPLE 1: Minimize f(θ) = 0.5 × (θ₁² + θ₂²)")
    print("="*60)
    
    def loss_fn(theta):
        return 0.5 * np.sum(theta ** 2)

    def grad_fn(theta):
        return theta

    initial_theta                 = np.array([5.0, -3.0])
    optimized_theta, loss_history = gradient_descent(loss_fn, grad_fn, initial_theta)
    print("Optimized Parameters:", np.round(optimized_theta, 4))
    print("Loss History:", [round(loss, 4) for loss in loss_history])
    
    # ================================================================
    # EXAMPLE 2: Single parameter optimization (1D)
    # ================================================================
    print("\n" + "="*60)
    print("EXAMPLE 2: Minimize f(x) = (x - 3)²")
    print("="*60)
    
    loss_fn = lambda x: (x - 3) ** 2  # type: ignore
    grad_fn = lambda x: 2 * (x - 3)   # type: ignore

    theta_opt, loss_hist = gradient_descent(loss_fn, grad_fn, theta_init=np.array(0.0), lr=0.01, steps=100)
    print("Optimized Parameters:", np.round(theta_opt, 4))
    print("Loss History:", [round(loss, 4) for loss in loss_hist])
