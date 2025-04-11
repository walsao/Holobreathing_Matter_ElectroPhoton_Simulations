# Install necessary packages if you haven't already:
# !pip install numpy matplotlib pillow

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
L = 400            # Length of the 1D field
T = 400            # Total time steps
dx = 1.0           # Space step
dt = 0.05          # Time step
v = 1.0            # Vacuum expectation value of breathing field
lambda_breath = 1.0  # Strength of the breathing potential

# Initialize breathing field phi and its time derivative phi_dot
phi = -v * np.ones(L)  # Field starts in "outside vacuum" (-v)
phi[L//2-20:L//2+20] = v  # Bubble in the middle (+v breathing phase)
phi_dot = np.zeros(L)  # No initial breathing motion

# Helper: calculate potential derivative dV/dphi
def dV_dphi(phi):
    return 4 * lambda_breath * phi * (phi**2 - v**2)

# Lists to store frames for animation
frames = []

# Main simulation loop
for t in range(T):
    # Finite difference for second spatial derivative (Laplacian)
    laplacian = (np.roll(phi, -1) - 2 * phi + np.roll(phi, 1)) / dx**2
    
    # Update acceleration from field equation
    phi_ddot = laplacian - dV_dphi(phi)
    
    # Symplectic integration (leapfrog method)
    phi_dot += phi_ddot * dt
    phi += phi_dot * dt
    
    # Store frame for visualization
    if t % 2 == 0:
        frames.append(phi.copy())

# Set up the animation
fig, ax = plt.subplots()
line, = ax.plot(np.linspace(0, L*dx, L), frames[0])
ax.set_ylim(-1.5*v, 1.5*v)
ax.set_title("Breathing Bubble Evolution (1D)")

def animate(i):
    line.set_ydata(frames[i])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50, blit=True)

# Save animation as GIF
ani.save('breathing_bubble_simulation.gif', writer='pillow')

print("✅ Breathing Bubble Simulation Complete! GIF saved as 'breathing_bubble_simulation.gif'!")
