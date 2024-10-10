import numpy as np
import matplotlib.pyplot as plt
import argparse

class DSMCSimulation:
    def __init__(self, n_particles, n_steps, T_tr_initial=380, T_rot_initial=180, Z_r=245, domain_size=6.4e-4, n_cells=10, sigma_collision=2.92e-10):
        # Simulation parameters
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.T_tr_initial = T_tr_initial
        self.T_rot_initial = T_rot_initial
        self.Z_r = Z_r
        self.p_inelastic = 1 / Z_r
        self.domain_size = domain_size
        self.n_cells = n_cells
        self.sigma_collision = sigma_collision
        self.k_B = 1.38e-23  # Boltzmann constant (J/K)
        self.m_H2 = 3.34e-26  # Mass of hydrogen molecule (kg)
        
        # Initialize arrays for positions, velocities, and energies
        self.positions = self.initialize_positions()
        self.velocities = self.initialize_velocities(self.T_tr_initial)
        self.rotational_energy = 0.5 * self.k_B * self.T_rot_initial * np.ones(self.n_particles)
        
        # Initialize spatial cells
        self.cells = np.zeros((self.n_cells, self.n_cells, self.n_cells), dtype=object)
        self.cell_size = self.domain_size / self.n_cells
        
        # Energy history for plotting
        self.translational_energy_history = []
        self.rotational_energy_history = []

    def initialize_velocities(self, T):
        """Initialize velocities based on Maxwell-Boltzmann distribution."""
        return np.random.normal(0, np.sqrt(self.k_B * T / self.m_H2), (self.n_particles, 3))

    def initialize_positions(self):
        """Initialize positions of particles randomly in the domain."""
        return np.random.rand(self.n_particles, 3) * self.domain_size

    def initialize_cells(self):
        """Initialize empty cells for particles."""
        self.cells = np.zeros((self.n_cells, self.n_cells, self.n_cells), dtype=object)
        for i in range(self.n_cells):
            for j in range(self.n_cells):
                for k in range(self.n_cells):
                    self.cells[i, j, k] = []

    def assign_to_cells(self):
        """Assign particles to cells based on their positions."""
        self.initialize_cells()  # Reset the cells before assignment
        for i in range(self.n_particles):
            cell_indices = (self.positions[i] // self.cell_size).astype(int)
            x, y, z = cell_indices
            self.cells[x, y, z].append(i)

    def compute_kinetic_energy(self, velocity):
        """Compute the kinetic energy of a particle."""
        return 0.5 * self.m_H2 * np.sum(velocity**2)

    def borgnakke_larsen_collision(self, kinetic_energy, rotational_energy):
        """Perform inelastic energy exchange between translational and rotational modes."""
        total_energy = kinetic_energy + rotational_energy
        new_kinetic_energy = np.random.rand() * total_energy
        new_rotational_energy = total_energy - new_kinetic_energy
        return new_kinetic_energy, new_rotational_energy

    def calculate_relative_velocity(self, vel1, vel2):
        """Calculate the relative velocity between two particles."""
        relative_velocity = vel1 - vel2
        return np.linalg.norm(relative_velocity)

    def perform_collision(self, idx1, idx2, max_relative_velocity):
        """Handle the collision between two particles."""
        velocity1, velocity2 = self.velocities[idx1], self.velocities[idx2]
        relative_velocity = self.calculate_relative_velocity(velocity1, velocity2)

        kinetic_energy1 = self.compute_kinetic_energy(velocity1)
        kinetic_energy2 = self.compute_kinetic_energy(velocity2)

        # Probability of collision based on relative velocity
        if np.random.rand() < relative_velocity / max_relative_velocity:
            # Inelastic collision
            if np.random.rand() < self.p_inelastic:
                kinetic_energy1, self.rotational_energy[idx1] = self.borgnakke_larsen_collision(kinetic_energy1, self.rotational_energy[idx1])
                kinetic_energy2, self.rotational_energy[idx2] = self.borgnakke_larsen_collision(kinetic_energy2, self.rotational_energy[idx2])

                self.velocities[idx1] *= np.sqrt(kinetic_energy1 / self.compute_kinetic_energy(velocity1))
                self.velocities[idx2] *= np.sqrt(kinetic_energy2 / self.compute_kinetic_energy(velocity2))

    def max_relative_velocity_in_cell(self, particles_in_cell):
        """Calculate the maximum relative velocity among all pairs in a cell."""
        max_rel_velocity = 0
        for i in range(len(particles_in_cell)):
            for j in range(i + 1, len(particles_in_cell)):
                idx1, idx2 = particles_in_cell[i], particles_in_cell[j]
                rel_velocity = self.calculate_relative_velocity(self.velocities[idx1], self.velocities[idx2])
                if rel_velocity > max_rel_velocity:
                    max_rel_velocity = rel_velocity
        return max_rel_velocity

    def calculate_total_energy(self):
        """Calculate the total translational and rotational energy in the system."""
        total_kinetic_energy = np.sum([self.compute_kinetic_energy(vel) for vel in self.velocities])
        total_rotational_energy = np.sum(self.rotational_energy)
        return total_kinetic_energy, total_rotational_energy

    def bl_dsmc_step(self):
        """Perform one step of BL-DSMC simulation, including particle collisions within cells."""
        self.assign_to_cells()

        # Loop over cells to perform collisions within each cell
        for i in range(self.n_cells):
            for j in range(self.n_cells):
                for k in range(self.n_cells):
                    cell_particles = self.cells[i, j, k]
                    n_cell_particles = len(cell_particles)

                    if n_cell_particles < 2:
                        continue  # No collisions in a cell with less than 2 particles

                    # Calculate the maximum relative velocity in the cell
                    max_rel_velocity = self.max_relative_velocity_in_cell(cell_particles)

                    # Perform collisions within the cell
                    for _ in range(n_cell_particles // 2):
                        idx1, idx2 = np.random.choice(cell_particles, 2, replace=False)
                        self.perform_collision(idx1, idx2, max_rel_velocity)

    def run_simulation(self, mode='full'):
        """Main simulation loop."""
        # Reset energy history
        self.translational_energy_history = []
        self.rotational_energy_history = []

        # Run the simulation for the defined number of steps
        for step in range(self.n_steps):
            self.bl_dsmc_step()

            # Calculate total translational and rotational energy
            total_kinetic_energy, total_rotational_energy = self.calculate_total_energy()
            self.translational_energy_history.append(total_kinetic_energy)
            self.rotational_energy_history.append(total_rotational_energy)

            # For quick validation, print the energy at every few steps in test mode
            if mode == 'test' and step % 100 == 0:
                print(f"Step {step}: Translational Energy = {total_kinetic_energy}, Rotational Energy = {total_rotational_energy}")

        print("Simulation complete.")

    def plot_energy_relaxation(self, mode='full'):
        """Plot the energy relaxation over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.translational_energy_history, label="Translational Energy", color='b')
        plt.plot(self.rotational_energy_history, label="Rotational Energy", color='r')
        plt.title(f"Energy Relaxation in DSMC ({mode} mode)")
        plt.xlabel("Time Step")
        plt.ylabel("Energy (J)")
        plt.legend()
        plt.show()

# Example usage: Run the simulation for both full and test modes
def main(mode='full'):
    
    if mode == 'full': # Full simulation parameters
        n_steps_full = 1000
        n_particles_full = 50000
    
    elif mode == 'test': # Test simulation parameters
        n_steps_test = 20000
        n_particles_test = 1000

    # # Run full simulation
    # full_simulation = DSMCSimulation(n_particles=n_particles_full, n_steps=n_steps_full)
    # full_simulation.run_simulation(mode='full')
    # full_simulation.plot_energy_relaxation(mode='full')

    # Run test simulation for validation
    test_simulation = DSMCSimulation(n_particles=n_particles_test, n_steps=n_steps_test)
    test_simulation.run_simulation(mode='test')
    test_simulation.plot_energy_relaxation(mode='test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSMC Simulation")
    parser.add_argument("--mode", type=str, default='full', help="Simulation mode: full or test")
    args = parser.parse_args()

    main(args.mode)
