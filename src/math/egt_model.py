import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def replicator_dynamics(x, t):
    """
    Computes the derivative dx/dt for the replicator dynamics ODE.
    
    x: Proportion of the population playing 'Cross' (C)
    1-x: Proportion of the population playing 'Wait' (W)
    
    Payoff Matrix (Symmetric Game in Phase 2):
    U(C, C) = -25  (Step: -5, Collision: -20)
    U(C, W) = -5   (Step: -5)
    U(W, C) = -3   (Wait: -3)
    U(W, W) = -3   (Wait: -3)
    """
    # Expected fitness for each pure strategy
    f_C = x * (-25) + (1 - x) * (-5)
    f_W = x * (-3) + (1 - x) * (-3)
    
    # Average population fitness
    phi = x * f_C + (1 - x) * f_W
    
    # Replicator equation
    dxdt = x * (f_C - phi)
    return dxdt

def run_simulation():
    """
    Simulates the replicator dynamics using scipy's ODE solver.
    Produces a visualization of population convergence.
    """
    t = np.linspace(0, 10, 200)
    initial_conditions = [0.01, 0.1, 0.5, 0.9, 0.99]
    
    plt.figure(figsize=(10, 6))
    
    for x0 in initial_conditions:
        sol = odeint(replicator_dynamics, x0, t)
        plt.plot(t, sol[:, 0], label=f"Initial C% = {x0*100:.1f}%")
        
    plt.title("Evolutionary Game Theory: Replicator Dynamics for Intersection Game", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Proportion of Population Playing 'Cross' (C)", fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='ESS (All Wait / Collapse)')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig('egt_simulation.png')
    print("Simulation complete. Saved to 'egt_simulation.png'.")

if __name__ == "__main__":
    run_simulation()
