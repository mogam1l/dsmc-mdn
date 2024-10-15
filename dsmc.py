import numpy as np
import matplotlib.pyplot as plt
import argparse
import tqdm

import tf_keras
import tensorflow_probability as tfp


tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers


class DSMCSimulation:
    def __init__(self, n_particles, n_steps, time_step=1e-6, use_mdn=False, mdn_model=None, T_tr_initial=380, T_rot_initial=180, Z_r=245, domain_size=6.4e-4, n_cells=10, sigma_collision=2.92e-10):
        # Simulation parameters
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.time_step = time_step  # Set timestep (in seconds)
        self.T_tr_initial = T_tr_initial
        self.T_rot_initial = T_rot_initial
        self.Z_r = Z_r
        self.p_inelastic = 1 / Z_r
        self.domain_size = domain_size
        self.n_cells = n_cells
        self.sigma_collision = sigma_collision
        self.k_B = 1.38e-23  # Boltzmann constant (J/K)
        self.m_H2 = 3.34e-26  # Mass of hydrogen molecule (kg)

        # Option to use the MDN-based surrogate model
        self.use_mdn = use_mdn
        self.mdn_model = mdn_model  # The MDN model passed during initialization
        
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

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def inv_sigmoid(self, x):
        """Inverse sigmoid function."""
        return np.log((x) / (1 - x))

    def mdn_energy_exchange(self, pre_collisional_energies):
        """Use the trained MDN to predict post-collisional energies."""
        # Input format: [log(E_c), inv_sigmoid(eps_t), inv_sigmoid(eps_r1)]
        log_Ec, inv_eps_t, inv_eps_r1 = pre_collisional_energies
        input_data = np.array([[log_Ec, inv_eps_t, inv_eps_r1]])

        # Perform prediction
        predictions = self.mdn_model.predict(input_data, verbose=0)

        # Extract predictions and transform back
        inv_eps_tp, inv_eps_r1p = predictions[0]
        #Ec_post = np.exp(pre_collisional_energies[0])       #post total is precolissional total
        eps_t_post = self.sigmoid(inv_eps_tp)
        eps_r1_post = self.sigmoid(inv_eps_r1p)

        return eps_t_post, eps_r1_post

    def perform_collision(self, idx1, idx2, max_relative_velocity):
        """Handle the collision between two particles."""
        velocity1, velocity2 = self.velocities[idx1], self.velocities[idx2]
        relative_velocity = self.calculate_relative_velocity(velocity1, velocity2)

        kinetic_energy1 = self.compute_kinetic_energy(velocity1)
        kinetic_energy2 = self.compute_kinetic_energy(velocity2)
        rotational_energy1, rotational_energy2 = self.rotational_energy[idx1], self.rotational_energy[idx2]

        # If we use the MDN, we replace the BL energy exchange process
        if self.use_mdn:
            # Pre-collision energy fractions and total energy
            Etr_total_pre = kinetic_energy1 + kinetic_energy2
            Er_total_pre = rotational_energy1 + rotational_energy2
            E_total_pre = Etr_total_pre + Er_total_pre

            eps_t = Etr_total_pre / E_total_pre #fraction of total energy in translational energy of BOTH molecules
            eps_r1 = rotational_energy1 / (rotational_energy1 + rotational_energy2) #Fraction of rotational energy in molecule A

            pre_collisional_energies = [
                np.log(E_total_pre),
                self.inv_sigmoid(eps_t),
                self.inv_sigmoid(eps_r1)
            ]

            eps_t_post, eps_r1_post = self.mdn_energy_exchange(pre_collisional_energies)
            Ec_post = E_total_pre

            total_postcolission_tr = Ec_post * eps_t_post
            total_postcolission_r = Ec_post * (1 - eps_t_post)

           
            # kinetic_energy1 = eps_t_post * Ec_post #Assumed translational energy?
            rotational_energy1 = eps_r1_post * (1 - eps_t_post) * Ec_post
            # kinetic_energy2 = (1 - eps_t_post) * Ec_post - rotational_energy1 #Assumed translational energy?
            rotational_energy2 = (1 - eps_r1_post) * (1 - eps_t_post) * Ec_post
            
            
            kinetic_energy1 = 0.5 * total_postcolission_tr #Assumed translational energy?
            # rotational_energy1 = total_postcolission_r * eps_r1_post     
            kinetic_energy2 = 0.5 * total_postcolission_tr #Assumed translational energy?
              

        else:
            # Default to BL model
            if np.random.rand() < self.p_inelastic:
                kinetic_energy1, rotational_energy1 = self.borgnakke_larsen_collision(kinetic_energy1, rotational_energy1)
                kinetic_energy2, rotational_energy2 = self.borgnakke_larsen_collision(kinetic_energy2, rotational_energy2)

        # Update the rotational energy
        self.rotational_energy[idx1] = rotational_energy1
        self.rotational_energy[idx2] = rotational_energy2

        # Rescale the velocities based on the new kinetic energy
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

    def calculate_relative_velocity(self, vel1, vel2):
        """Calculate the relative velocity between two particles."""
        relative_velocity = vel1 - vel2
        return np.linalg.norm(relative_velocity)

    def calculate_total_energy(self):
        """Calculate the total translational and rotational energy in the system."""
        total_kinetic_energy = np.sum([self.compute_kinetic_energy(vel) for vel in self.velocities])
        total_rotational_energy = np.sum(self.rotational_energy)
        return total_kinetic_energy, total_rotational_energy

    def update_positions(self):
        """Update particle positions based on velocity and timestep, applying periodic boundary conditions."""
        self.positions += self.velocities * self.time_step
        
        # Apply periodic boundary conditions
        self.positions = np.mod(self.positions, self.domain_size) # PERIODIC BOUNDARY CONDITIONS

    def dsmc_step(self):
        """Perform one step of BL-DSMC simulation, including particle collisions within cells and updating positions."""
        # Update particle positions based on velocities and timestep
        self.update_positions()

        # Re-assign particles to cells after position update
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
        with tqdm.tqdm(total=self.n_steps) as pbar:
            for step in range(self.n_steps):
                pbar.set_description(f"Step {step}")
                self.dsmc_step()

                # Calculate total translational and rotational energy
                total_kinetic_energy, total_rotational_energy = self.calculate_total_energy()
                self.translational_energy_history.append(total_kinetic_energy)
                self.rotational_energy_history.append(total_rotational_energy)

                # For quick validation, print the energy at every few steps in test mode
                if mode == 'test' and step % 100 == 0:
                    print(f"Step {step}: Translational Energy = {total_kinetic_energy}, Rotational Energy = {total_rotational_energy}")

                pbar.update(1)

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


if __name__ == "__main__":
    # Load the trained MDN model here (assuming TensorFlow model)
    parser = argparse.ArgumentParser(description="DSMC Simulation")

    parser.add_argument("--mdn", type=str, default=None, help="Path to the trained MDN model")
    parser.add_argument("--n_particles", type=int, default=500, help="Number of particles")
    parser.add_argument("--n_steps", type=int, default=20000, help="Number of steps")
    parser.add_argument("--time_step", type=float, default=1e-6, help="Timestep in seconds")

    args = parser.parse_args()

    if args.mdn: ##--mdn path
        print("Using MDN model for energy exchange.")
        use_mdn = True
        
        def build_model(NGAUSSIANS, ACTIVATION, NNEURONS):
            
            event_shape = [2]
            num_components = NGAUSSIANS
            params_size = tfpl.MixtureSameFamily.params_size(num_components,
                            component_params_size=tfpl.IndependentNormal.params_size(event_shape))

            negloglik = lambda y, p_y: -p_y.log_prob(y)

            model = tf_keras.models.Sequential([
                tf_keras.layers.Dense(NNEURONS, activation=ACTIVATION),
                tf_keras.layers.Dense(params_size, activation=None),
                tfpl.MixtureSameFamily(num_components, tfpl.IndependentNormal(event_shape)),
            ])
            
            model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate = 1e-4), loss=negloglik)

            return model


        mdn_model = build_model(20, 'relu', 8) #settings have to match otherwise the loaded weights won't wrork.

        #needed initialization step, probably determines the input size from here.
        mdn_model(np.ones((3,3)))

        mdn_model.load_weights(args.mdn)
        mdn_model.summary()
        
    else:
        use_mdn = False
        mdn_model = None

    dsmc = DSMCSimulation(n_particles=args.n_particles, n_steps=args.n_steps, time_step=args.time_step, use_mdn=use_mdn, mdn_model=mdn_model)
    dsmc.run_simulation()
    dsmc.plot_energy_relaxation()
