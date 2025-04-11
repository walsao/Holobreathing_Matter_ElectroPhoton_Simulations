# Install necessary packages if you haven't already:
# !pip install numpy matplotlib pillow

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
Lx, Ly = 100, 100   # Grid size
T = 300             # Total time steps
dx = 1.0            # Space step
dt = 0.05           # Time step
v = 1.0             # Breathing field vacuum amplitude
lambda_breath = 1.0  # Breathing potential strength

# Create 2D grid
x = np.linspace(0, Lx*dx, Lx)
y = np.linspace(0, Ly*dx, Ly)
X, Y = np.meshgrid(x, y)

# Initialize breathing field phi and its time derivative phi_dot
phi = -v * np.ones((Lx, Ly))  # Outside vacuum (-v)
phi_dot = np.zeros((Lx, Ly))  # No initial breathing

# Create a circular breathing bubble at center
r = np.sqrt((X - Lx*dx/2)**2 + (Y - Ly*dx/2)**2)
bubble_radius = 15.0
phi[r < bubble_radius] = v  # Inside bubble (+v)

# Helper: calculate potential derivative dV/dphi
def dV_dphi(phi):
    return 4 * lambda_breath * phi * (phi**2 - v**2)

# Lists to store frames for animation
frames = []

# Main simulation loop
for t in range(T):
    # 2D Laplacian
    laplacian = (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) -
        4 * phi
    ) / dx**2
    
    # Update acceleration from field equation
    phi_ddot = laplacian - dV_dphi(phi)
    
    # Symplectic integration (leapfrog method)
    phi_dot += phi_ddot * dt
    phi += phi_dot * dt
    
    # Store frame for visualization
    if t % 3 == 0:
        frames.append(phi.copy())

# Set up the animation
fig, ax = plt.subplots()
cax = ax.imshow(frames[0], cmap='plasma', origin='lower', vmin=-1.5*v, vmax=1.5*v)
fig.colorbar(cax)
ax.set_title("2D Breathing Bubble Evolution")

def animate(i):
    cax.set_array(frames[i])
    return cax,

ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50, blit=False)

# Save animation as a GIF
ani.save('breathing_bubble_2D.gif', writer='pillow')

print("✅ 2D Breathing Bubble Simulation Complete! GIF saved as 'breathing_bubble_2D.gif'!")
