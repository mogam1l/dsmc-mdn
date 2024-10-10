import numpy as np
import tqdm

# Constants
k_B = 1.38e-23  # Boltzmann constant (J/K)
m_H2 = 3.34e-26  # Mass of hydrogen molecule (kg)
Z_r = 245  # Rotational relaxation number
p_inelastic = 1 / Z_r  # Inelastic collision probability
domain_size = 6.4e-4  # m^2 (size of the domain)
n_cells = 10  # Number of cells in each dimension (for a grid)
sigma_collision = 2.92e-10  # Molecular diameter (m)

# Simulation Parameters
n_particles = 50000  # Number of DSMC particles
T_tr_initial = 380  # Initial translational temperature in Kelvin
T_rot_initial = 180  # Initial rotational temperature in Kelvin
T_eq = 300  # Equilibrium temperature
time_step = 1e-6  # Time step (s)

# Initialize the cells
cells = np.zeros((n_cells, n_cells, n_cells), dtype=object)

def initialize_velocities(T, n_particles):
    """Initialize velocities based on Maxwell-Boltzmann distribution."""
    return np.random.normal(0, np.sqrt(k_B * T / m_H2), (n_particles, 3))

def initialize_positions(domain_size, n_particles):
    """Initialize positions of particles randomly in the domain."""
    return np.random.rand(n_particles, 3) * domain_size

def initialize_cells():
    """Initialize empty cells for particles."""
    global cells
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                cells[i, j, k] = []

def assign_to_cells(positions, n_cells, cell_size):
    """Assign particles to cells based on their positions."""
    initialize_cells()  # Reset the cells before assignment
    for i in range(n_particles):
        cell_indices = (positions[i] // cell_size).astype(int)
        x, y, z = cell_indices
        cells[x, y, z].append(i)

def compute_kinetic_energy(velocity):
    """Compute the kinetic energy of a particle."""
    return 0.5 * m_H2 * np.sum(velocity**2)

def borgnakke_larsen_collision(kinetic_energy, rotational_energy):
    """Perform inelastic energy exchange between translational and rotational modes."""
    total_energy = kinetic_energy + rotational_energy
    new_kinetic_energy = np.random.rand() * total_energy
    new_rotational_energy = total_energy - new_kinetic_energy
    return new_kinetic_energy, new_rotational_energy

def calculate_relative_velocity(vel1, vel2):
    """Calculate the relative velocity between two particles."""
    relative_velocity = vel1 - vel2
    relative_speed = np.linalg.norm(relative_velocity)
    return relative_speed

def perform_collision(idx1, idx2, velocities, rotational_energy, max_relative_velocity):
    """Handle the collision between two particles."""
    velocity1, velocity2 = velocities[idx1], velocities[idx2]
    relative_velocity = calculate_relative_velocity(velocity1, velocity2)

    kinetic_energy1 = compute_kinetic_energy(velocity1)
    kinetic_energy2 = compute_kinetic_energy(velocity2)

    # Probability of collision based on relative velocity
    if np.random.rand() < relative_velocity / max_relative_velocity:
        # Inelastic collision
        if np.random.rand() < p_inelastic:
            kinetic_energy1, rotational_energy[idx1] = borgnakke_larsen_collision(kinetic_energy1, rotational_energy[idx1])
            kinetic_energy2, rotational_energy[idx2] = borgnakke_larsen_collision(kinetic_energy2, rotational_energy[idx2])

            velocities[idx1] *= np.sqrt(kinetic_energy1 / compute_kinetic_energy(velocity1))
            velocities[idx2] *= np.sqrt(kinetic_energy2 / compute_kinetic_energy(velocity2))

def max_relative_velocity_in_cell(particles_in_cell, velocities):
    """Calculate the maximum relative velocity among all pairs in a cell."""
    max_rel_velocity = 0
    for i in range(len(particles_in_cell)):
        for j in range(i + 1, len(particles_in_cell)):
            idx1, idx2 = particles_in_cell[i], particles_in_cell[j]
            rel_velocity = calculate_relative_velocity(velocities[idx1], velocities[idx2])
            if rel_velocity > max_rel_velocity:
                max_rel_velocity = rel_velocity
    return max_rel_velocity

def bl_dsmc_step(positions, velocities, rotational_energy, cell_size):
    """Perform one step of BL-DSMC simulation, including particle collisions within cells."""
    assign_to_cells(positions, n_cells, cell_size)

    # Loop over cells to perform collisions within each cell
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                cell_particles = cells[i, j, k]
                n_cell_particles = len(cell_particles)

                if n_cell_particles < 2:
                    continue  # No collisions in a cell with less than 2 particles

                # Calculate the maximum relative velocity in the cell
                max_rel_velocity = max_relative_velocity_in_cell(cell_particles, velocities)

                # Perform collisions within the cell
                for _ in range(n_cell_particles // 2):
                    idx1, idx2 = np.random.choice(cell_particles, 2, replace=False)
                    perform_collision(idx1, idx2, velocities, rotational_energy, max_rel_velocity)

def run_simulation(n_steps):
    """Main simulation loop."""
    # Initialize particle velocities and positions
    velocities = initialize_velocities(T_tr_initial, n_particles)
    rotational_energy = 0.5 * k_B * T_rot_initial * np.ones(n_particles)
    positions = initialize_positions(domain_size, n_particles)
    cell_size = domain_size / n_cells

    # Run the simulation for the defined number of steps
    with tqdm.tqdm(total=n_steps) as pbar:
        for step in range(n_steps):
            bl_dsmc_step(positions, velocities, rotational_energy, cell_size)
            pbar.update(1)


    print("Simulation complete.")
    return velocities, rotational_energy

# Example usage: Run the simulation for 1000 steps
def main():
    n_steps = 1000
    velocities, rotational_energy = run_simulation(n_steps)
    print("Final velocities:\n", velocities)
    print("Final rotational energies:\n", rotational_energy)

if __name__ == "__main__":
    main()
