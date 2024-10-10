import numpy as np
import tqdm

# Constants
k_B = 1.38e-23  # Boltzmann constant (J/K)
m_H2 = 3.34e-26  # Mass of hydrogen molecule (kg)
Z_r = 245  # Rotational relaxation number
p_inelastic = 1 / Z_r  # Inelastic collision probability

# Simulation Parameters
n_particles = 50000  # Number of DSMC particles
T_tr_initial = 380  # Initial translational temperature in Kelvin
T_rot_initial = 180  # Initial rotational temperature in Kelvin
T_eq = 300  # Equilibrium temperature
time_step = 1e-6  # Time step (s)

# Initialize velocities based on Maxwell-Boltzmann distribution
def initialize_velocities(T, n_particles):
    return np.random.normal(0, np.sqrt(k_B * T / m_H2), (n_particles, 3))

# Initialize translational and rotational energy
velocities = initialize_velocities(T_tr_initial, n_particles)
rotational_energy = 0.5 * k_B * T_rot_initial * np.ones(n_particles)

# Functions for energy exchange
def compute_kinetic_energy(velocity):
    """Compute the kinetic energy of a particle."""
    return 0.5 * m_H2 * np.sum(velocity**2)

def borgnakke_larsen_collision(kinetic_energy, rotational_energy):
    """Perform inelastic energy exchange between translational and rotational modes."""
    total_energy = kinetic_energy + rotational_energy
    new_kinetic_energy = np.random.rand() * total_energy
    new_rotational_energy = total_energy - new_kinetic_energy
    return new_kinetic_energy, new_rotational_energy

def bl_dsmc_step():
    """Perform one step of BL-DSMC simulation."""
    with tqdm.tqdm(total=n_particles // 2) as pbar:
        for i in range(n_particles // 2):
            
            idx1, idx2 = np.random.choice(n_particles, 2, replace=False)
            velocity1, velocity2 = velocities[idx1], velocities[idx2]
            relative_velocity = velocity1 - velocity2
            relative_speed = np.linalg.norm(relative_velocity)
            
            kinetic_energy1 = compute_kinetic_energy(velocity1)
            kinetic_energy2 = compute_kinetic_energy(velocity2)
            
            # Randomly perform inelastic collision
            if np.random.rand() < p_inelastic:
                kinetic_energy1, rotational_energy[idx1] = borgnakke_larsen_collision(kinetic_energy1, rotational_energy[idx1])
                kinetic_energy2, rotational_energy[idx2] = borgnakke_larsen_collision(kinetic_energy2, rotational_energy[idx2])
                
                velocities[idx1] *= np.sqrt(kinetic_energy1 / compute_kinetic_energy(velocity1))
                velocities[idx2] *= np.sqrt(kinetic_energy2 / compute_kinetic_energy(velocity2))
            pbar.update(1)

# Run the simulation
n_steps = 1000
with tqdm.tqdm(total=n_steps) as pbar:
    for step in range(n_steps):
        pbar.set_description(f"Step {step + 1}/{n_steps}")
        bl_dsmc_step()
        pbar.update(1)

print("Simulation complete.")
