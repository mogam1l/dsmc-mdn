import numpy as np
import matplotlib.pyplot as plt
import argparse
import tqdm
from scipy.special import expit, logit, softmax
from scipy.stats import norm
import os
import csv

import tf_keras
import tensorflow_probability as tfp


tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

np.random.seed(1)


class DSMCSimulation:
    def __init__(self, n_particles, n_steps, time_step=1e-6, mdn_model=None, T_tr_initial=167, T_rot_initial=1000, Z_r=245, domain_size=6.4e-4, n_cells=10, sigma_collision=2.92e-10):
        # Simulation parameters
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.T_tr_initial = T_tr_initial
        self.T_rot_initial = T_rot_initial
        self.Z_r = Z_r
        self.p_inelastic = 1/self.Z_r  # Inelastic collision probability
        self.domain_size = domain_size
        self.n_cells = n_cells
        self.sigma_collision = sigma_collision


        self.k_B = 1.38e-23  # Boltzmann constant (J/K)
        self.m_H2 = 3.34e-26  # Mass of hydrogen molecule (kg)
        self.density = 0.9  # Density of particles (kg/m^3)
        self.omega = 0.5  # Collision model parameter
        
        # Degrees of freedom
        self.dof_trans = 3  # 3 translational degrees of freedom
        self.dof_rot = 2    # 2 rotational degrees of freedom for diatomic molecules

        # v_init = np.sqrt(3*boltz*T/mass)
        self.v_init = np.sqrt(3 * self.k_B * self.T_tr_initial / self.m_H2)  # Initial velocity
        #tau = 0.2*(L/ncell)/v_init
        self.time_step = 0.2 * (self.domain_size / self.n_cells) / self.v_init   # Set timestep (in seconds)
        # eff_num = density/mass * L**3 /npart
        self.eff_num = self.density/self.m_H2 * domain_size**3 / n_particles
        # coeff = 0.5*eff_num*np.pi*diam**2*tau/(L**3/ncell)
        self.coeff = 0.5 * self.eff_num * np.pi * self.sigma_collision**2 * self.time_step / (self.domain_size**3 / self.n_cells)

        # Option to use the MDN-based surrogate model
        if mdn_model is not None:
            print("Using MDN model for energy exchange.")
            self.use_mdn = True
            self.mdn_model = mdn_model  # The MDN model passed during initialization
            self.w1 = self.mdn_model.get_weights()[0]
            self.b1 = self.mdn_model.get_weights()[1]
            self.w2 = self.mdn_model.get_weights()[2]
            self.b2 = self.mdn_model.get_weights()[3]
            self.Ngauss = Ngauss
        else:
            print("Using regular Larsen-Borgnakke model for energy exchange.")
            self.use_mdn = False
        
        # Initialize arrays for positions, velocities, and energies
        self.positions = self.initialize_positions()
        self.velocities = self.initialize_velocities(self.T_tr_initial)
        self.rotational_energy = -np.log(np.random.rand(self.n_particles)) * self.k_B * self.T_rot_initial
        
        # Initialize spatial cells
        self.cells = np.zeros((self.n_cells, self.n_cells, self.n_cells), dtype=object)
        self.cell_size = self.domain_size / self.n_cells
        self.volume_cell = self.cell_size ** 3  # Volume of each cell
        
        # Energy history for plotting
        self.translational_energy_history = []
        self.rotational_energy_history = []
        self.total_energy_history = []
        self.elastic_collisions = 0
        self.inelastic_collisions = 0
        self.rejected_collisions = 0
        self.current_step = 0

        self.initialize_logger()
        
    def initialize_logger(self):
        """Initialize the collision logger."""
        if self.use_mdn:
            model_type = "MDN"
        else:
            model_type = "BL"
        # name file according to dsmc parameters
        self.log_file = open(f"dsmc_{model_type}_N:{self.n_particles}_steps:{self.n_steps}_Ttr:{self.T_tr_initial}_Trot:{self.T_rot_initial}_Zr:{self.Z_r}_domain:{self.domain_size}_cells{self.n_cells}_sigma{self.sigma_collision}.csv", 'w', newline='')
        self.log_writer = csv.writer(self.log_file)
        # Write headers
        self.log_writer.writerow(['time_step','collision_type', 'b_parameter', 'E_total_pre', 'E_total_post', 'E_trans_pre', 'E_trans_post', 'E_rot_pre', 'E_rot_post'])

    def close_logger(self):
        """Close the collision logger file."""
        self.log_file.close()

    def initialize_velocities(self, T):
        """Initialize velocities based on Maxwell-Boltzmann distribution."""
        return np.sqrt(2 * self.k_B * T / self.m_H2) * np.sin(2 * np.pi * np.random.rand(self.n_particles, 3)) * np.sqrt(-np.log(np.random.rand(self.n_particles, 3)))

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

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def inv_sigmoid(self, x):
        """Inverse sigmoid function with clamping to avoid division by zero."""
        epsilon = 1e-9  # Small value to avoid log(0) or division by zero
        x = np.clip(x, epsilon, 1 - epsilon)  # Ensure x is within (0, 1)
        return np.log(x / (1 - x))
    
    def softplus(self, x):
        """Compute the Softplus function."""
        return np.log1p(np.exp(x))  # Using log1p for numerical stability

    def inverse_softplus(self, x):
        """Compute the Inverse Softplus function."""
        if np.any(x <= 0):
            raise ValueError("Inverse Softplus is defined only for y > 0")
        return np.log(np.expm1(x))  # Using expm1 for numerical stability

    def mdn_energy_exchange_new(self, Ec, eps_t, eps_rP):
        
        # Ensure inputs are numpy arrays
        Ec = np.asarray(Ec)
        eps_t = np.asarray(eps_t)
        eps_rP = np.asarray(eps_rP)
    
        # Constructing input vector for MDN
        input_vec = np.vstack((np.log(Ec), logit(eps_t), logit(eps_rP))).T  # Shape: (N, 3)
    
        # Initialize output matrices
        output_HL = np.maximum(0, np.dot(input_vec, self.w1) + self.b1)  # ReLU activation
    
        output_OL = np.dot(output_HL, self.w2) + self.b2  # Linear activation
    
        N = len(output_OL[:,0])
        # Initialize arrays for GMM parameters
        weights = np.zeros((N, self.Ngauss))
        mu_eps_t = np.zeros((N, self.Ngauss))
        mu_eps_rP = np.zeros((N, self.Ngauss))
        sigma_eps_t = np.zeros((N, self.Ngauss))
        sigma_eps_rP = np.zeros((N, self.Ngauss))
        
        # Extracting parameters from the output layer
        for i in range(self.Ngauss):
            weights[:, i] = output_OL[:, i]
            mu_eps_t[:, i] = output_OL[:, self.Ngauss + (i * 4)]
            mu_eps_rP[:, i] = output_OL[:, (self.Ngauss + 1) + (i * 4)]
            sigma_eps_t[:, i] = output_OL[:, (self.Ngauss + 2) + (i * 4)]
            sigma_eps_rP[:, i] = output_OL[:, (self.Ngauss + 3) + (i * 4)]
        
        # Softmax for weights #to keep them positive
        weights = softmax(weights)
    
        # Softplus for standard deviations#This seems hacky, does TF do this?
        sigma_eps_t = self.softplus(sigma_eps_t)  # Softplus
        sigma_eps_rP = self.softplus(sigma_eps_rP) # Softplus
        
    
        # Sampling from mixture distribution
        # N = input_vec.shape[0]
        Un = np.random.rand(N)  # Draw random numbers for each input
        mu_t, mu_rP, sigma_t, sigma_rP = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    
        # Loop through each instance to sample from the mixture model
        for i in range(N):
            weights_sum_vec = np.zeros(self.Ngauss)
            weights_sum_vec[0] = weights[i, 0]
            for j in range(1, self.Ngauss):
                weights_sum_vec[j] = weights[i, j] + weights_sum_vec[j - 1]
    
            for j in range(self.Ngauss):
                if Un[i] < weights_sum_vec[j]:
                    mu_t[i] = mu_eps_t[i, j]
                    mu_rP[i] = mu_eps_rP[i, j]
                    sigma_t[i] = sigma_eps_t[i, j]
                    sigma_rP[i] = sigma_eps_rP[i, j]
                    break
        
        # Sampled normal distribution for eps_t and eps_rP
        eps_t_p = norm.rvs(loc=mu_t, scale=sigma_t)  # Fraction of translational energy (post-collision)
        eps_rP_p = norm.rvs(loc=mu_rP, scale=sigma_rP)  # Fraction of rotational energy in molecule P (post-collision)
    
        # Post-processing fractions back to [0, 1] range with Sigmoid function
        eps_t_p = expit(eps_t_p)
        eps_rP_p = expit(eps_rP_p)
    
        return eps_t_p, eps_rP_p#, translationalEnergy, ERotP, ERotQ
    
    def update_energy_and_velocity(self, idx1, idx2, E_total_pre, energy_fraction_trans, energy_fraction_rot1, CM_velocity):
        E_trans_post = E_total_pre * energy_fraction_trans
        E_rot_post = E_total_pre - E_trans_post
 
        self.rotational_energy[idx1] = E_rot_post * energy_fraction_rot1
        self.rotational_energy[idx2] = E_rot_post - self.rotational_energy[idx1]

        # Correct calculation of relative speed
        relative_speed = np.sqrt(4 * E_trans_post / self.m_H2)
        new_relative_velocity = self.random_unit_vector() * relative_speed

        # Update velocities
        self.velocities[idx1] = CM_velocity + 0.5 * new_relative_velocity
        self.velocities[idx2] = CM_velocity - 0.5 * new_relative_velocity

        # Energy conservation check
        E_trans_post_check = 0.25 * self.m_H2 * np.sum(new_relative_velocity**2)
        E_rot_post_check = self.rotational_energy[idx1] + self.rotational_energy[idx2]
        E_total_post = E_trans_post_check + E_rot_post_check

        self.log_writer.writerow([self.current_step*self.time_step, "inelastic", self.b_parameter, E_total_pre, E_total_post, self.E_trans_pre, E_trans_post, self.E_rot_pre, E_rot_post])

        assert np.isclose(E_total_pre, E_total_post, atol=1e-10), "Energy conservation violated!"
    
    def perform_collision(self, idx1, idx2 , max_rel_velocity):
        """Handle the collision between two particles using the regular Larsen-Borgnakke model."""
        velocity1, velocity2 = self.velocities[idx1], self.velocities[idx2]
        relative_velocity = self.calculate_relative_velocity(velocity1, velocity2)
        CM_velocity = 0.5 * (velocity1 + velocity2)
        self.b_parameter = self.compute_b_parameter(idx1,idx2)
        # Total energy before collision

        self.E_trans_pre = 0.25 * self.m_H2 * np.sum(relative_velocity**2)
        self.E_rot_pre = self.rotational_energy[idx1] + self.rotational_energy[idx2]
        self.E_total_pre = self.E_trans_pre + self.E_rot_pre


        # Compute collision probability using fixed v_rel_max
        collision_prob = np.linalg.norm(relative_velocity) / max_rel_velocity
        if np.random.rand() > collision_prob:
            self.rejected_collisions += 1
            return  # Skip collision if it does not happen based on collision probability

        # Test for elastic collision for the collision pair
        if np.random.rand() > self.p_inelastic:  # If true: Elastic collision will happen
            self.elastic_collisions += 1

            # Isotropic scattering (randomize the relative velocity direction)
            relative_speed = np.linalg.norm(relative_velocity)
            new_relative_velocity = self.random_unit_vector() * relative_speed

            # Update velocities
            self.velocities[idx1] = CM_velocity + 0.5 * new_relative_velocity
            self.velocities[idx2] = CM_velocity - 0.5 * new_relative_velocity

            E_trans_post = 0.25 * self.m_H2 * np.sum(new_relative_velocity**2)
            E_rot_post = self.E_rot_pre
            E_total_post = E_trans_post + E_rot_post

            #  self.log_writer.writerow(['time_step','collision_type', 'b_parameter', 'E_total_pre', 'E_total_post', 'E_trans_pre', 'E_trans_post', 'E_rot_pre', 'E_rot_post'])
            self.log_writer.writerow([self.current_step*self.time_step, "elastic", self.b_parameter, self.E_total_pre, E_total_post, self.E_trans_pre, E_trans_post, self.E_rot_pre, E_rot_post])
            return
        
        if self.use_mdn == False:  # Inelastic collision using regular Larsen-Borgnakke model
            self.inelastic_collisions += 1

            # Redistribute total energy among translational and rotational modes
            energy_fraction_trans = np.random.rand()
            E_tr_probability = ((self.dof_rot + 0.5 - self.omega)/(1.5 - self.omega) * energy_fraction_trans)**(1.5 - self.omega) * ((self.dof_rot + 0.5 - self.omega)/(self.dof_rot - 1) * (1 - energy_fraction_trans))**(self.dof_rot - 1)
            while E_tr_probability < np.random.rand():
                energy_fraction_trans = np.random.rand()
                E_tr_probability = ((self.dof_rot + 0.5 - self.omega)/(1.5 - self.omega) * energy_fraction_trans)**(1.5 - self.omega) * ((self.dof_rot + 0.5 - self.omega)/(self.dof_rot - 1) * (1 - energy_fraction_trans))**(self.dof_rot - 1)
            
            # Distribute rotational energy equally between the two molecules
            energy_fraction_rot1 = np.random.rand()
            E_rot1_probability = 2**(self.dof_rot - 2) * energy_fraction_rot1**(self.dof_rot/2 - 1) * (1 - energy_fraction_rot1)**(self.dof_rot/2 - 1)
            while E_rot1_probability < np.random.rand():
                energy_fraction_rot1 = np.random.rand()
                E_rot1_probability = 2**(self.dof_rot - 2) * energy_fraction_rot1**(self.dof_rot/2 - 1) * (1 - energy_fraction_rot1)**(self.dof_rot/2 - 1)

            #  update_energy_and_velocity(self, idx1, idx2, E_total_pre, energy_fraction_trans, energy_fraction_rot1, CM_velocity):
            self.update_energy_and_velocity(idx1, idx2, self.E_total_pre, energy_fraction_trans, energy_fraction_rot1, CM_velocity)

            
        elif self.use_mdn == True: # Inelastic collision using MDN-based surrogate model
            self.inelastic_collisions += 1
            
            eps_t_pre = self.E_trans_pre /self. E_total_pre
            eps_r1_pre = self.rotational_energy[idx1] / self.E_rot_pre
            
            energy_fraction_trans, energy_fraction_rot1 = self.mdn_energy_exchange_new(self.E_total_pre, eps_t_pre, eps_r1_pre)
            
            #  update_energy_and_velocity(self, idx1, idx2, E_total_pre, energy_fraction_trans, energy_fraction_rot1, CM_velocity):
            self.update_energy_and_velocity(idx1, idx2, self.E_total_pre, energy_fraction_trans, energy_fraction_rot1, CM_velocity)
                


    def random_unit_vector(self):
        """Generate a random unit vector uniformly distributed over the sphere."""
        theta = np.pi * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z])

    def max_relative_velocity_in_cell(self, particles_in_cell):
        """Calculate the maximum relative velocity (magnitude) among all pairs in a cell."""
        max_rel_velocity = 0
        for i in range(len(particles_in_cell)):
            for j in range(i + 1, len(particles_in_cell)):
                idx1, idx2 = particles_in_cell[i], particles_in_cell[j]
                rel_velocity = self.calculate_relative_velocity(self.velocities[idx1], self.velocities[idx2])

                # Compare the magnitude (norm) of the relative velocity vectors
                rel_velocity_magnitude = np.linalg.norm(rel_velocity)
                if rel_velocity_magnitude > max_rel_velocity:
                    max_rel_velocity = rel_velocity_magnitude

        return max_rel_velocity

    def calculate_relative_velocity(self, vel1, vel2):
        """Calculate the relative velocity between two particles."""
        relative_velocity = vel1 - vel2
        return relative_velocity

    def calculate_total_energy(self):
        """Calculate the total translational and rotational energy in the system."""
        total_kinetic_energy = np.sum([self.compute_kinetic_energy(vel) for vel in self.velocities])
        total_rotational_energy = np.sum(self.rotational_energy)
        total_energy = (total_kinetic_energy  + total_rotational_energy)/1
        return total_kinetic_energy, total_rotational_energy , total_energy

    def update_positions(self):
        """Update particle positions based on velocity and timestep, applying periodic boundary conditions."""
        self.positions += self.velocities * self.time_step
        
        # Apply periodic boundary conditions
        self.positions = np.mod(self.positions, self.domain_size) # PERIODIC BOUNDARY CONDITIONS
        
    def compute_b_parameter(self, id1, id2):
        """Compute b parameter based on the direction of the relative velocity vector and the particle positions"""
        time_of_collision = (self.positions[id1] - self.positions[id2])/(self.velocities[id1] - self.velocities[id2])
        pos1 = self.positions[id1] + self.velocities[id1] * time_of_collision
        pos2 = self.positions[id2] + self.velocities[id2] * time_of_collision
        b_parameter = np.linalg.norm(pos1 - pos2)
        return b_parameter

    def plot_positions(self):
        """Plot the positions of particles in 3D."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Particle Positions")
        plt.show()

    def dsmc_step(self):
        """Perform one step of BL-DSMC simulation, including particle collisions within cells and updating positions."""
        self.current_step += 1


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

                    # Calculate number of candidate collision pairs to be selected (based on the original code)
                    # select = coeff*number*(number-1)*crmax[jcell] 
                    n_candidate_pairs = int(self.coeff * n_cell_particles * (n_cell_particles - 1) * max_rel_velocity)
                    # print(f"Cell ({i}, {j}, {k}): {n_cell_particles} particles, {n_candidate_pairs} candidate pairs") 
                    # Print all the parameters for debugging

                    # Perform candidate collisions
                    for _ in range(n_candidate_pairs):
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
                total_kinetic_energy, total_rotational_energy, total_energy = self.calculate_total_energy()
                self.translational_energy_history.append(total_kinetic_energy)
                self.rotational_energy_history.append(total_rotational_energy)
                self.total_energy_history.append(total_energy)


                # For quick validation, print the energy at every few steps in test mode
                if mode == 'test' and step % 100 == 0:
                    print(f"Step {step}: Translational Energy = {total_kinetic_energy}, Rotational Energy = {total_rotational_energy}")

                pbar.update(1)

        print("Simulation complete.")
        self.close_logger()

    def plot_energy_relaxation(self, mode='full'):
        """Plot the energy relaxation over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.translational_energy_history, label="Translational Energy", color='b')
        plt.plot(self.rotational_energy_history, label="Rotational Energy", color='r')
        plt.plot(self.total_energy_history, label="Total Energy", color='k')
        plt.title(f"Energy Relaxation in DSMC ({mode} mode)")
        plt.xlabel("Time Step")
        plt.ylabel("Energy (K)")
        plt.legend()
        plt.show()
        
    def plot_energy_relaxation_T(self, mode='full'):
        self.T_tr = np.array(self.translational_energy_history) / (0.5 * self.dof_trans * self.n_particles * self.k_B)
        self.T_rot = np.array(self.rotational_energy_history) / (0.5 * self.dof_rot * self.n_particles * self.k_B)
        self.T_total = np.array(self.total_energy_history) / (0.5 * (self.dof_trans + self.dof_rot) * self.n_particles * self.k_B)
        
        """Plot the energy relaxation over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.T_tr, label="Translational Temperature", color='b')
        plt.plot(self.T_rot, label="Rotational Temperature", color='r')
        plt.plot(self.T_total, label="System Temperature", color='k')
        plt.title(f"Energy Relaxation in DSMC ({mode} mode)")
        plt.xlabel("Time Step")
        plt.ylabel("Energy (K)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Load the trained MDN model here (assuming TensorFlow model)
    parser = argparse.ArgumentParser(description="DSMC Simulation")

    parser.add_argument("--mdn_model", type=str, default=None, help="Path to the trained MDN model")
    parser.add_argument("--n_particles", type=int, default=5000, help="Number of particles")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--time_step", type=float, default=1e-6, help="Timestep in seconds")

    args = parser.parse_args()

    if args.mdn_model: ##--mdn path
        print(f"Loading MDN model from {args.mdn_model}")
        use_mdn = True
        
        # Disable oneDNN optimizations
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #suppresses rounding error messages
        
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

        # DN Properties
        Ngauss = 20  # Number of Gaussians
        Nneurons = 8  # Number of neurons in hidden layer
        # Nneurons_OL = Ngauss * 2 * 2 + Ngauss  # Number of neurons in output layer        

        mdn_model = build_model(Ngauss, 'relu', Nneurons) #settings have to match otherwise the loaded weights won't wrork.

        #needed initialization step, probably determines the input size from here.
        mdn_model(np.ones((1,3)))

        mdn_model.load_weights(args.mdn_model)
        mdn_model.summary()
        
        w1 = mdn_model.get_weights()[0]
        b1 = mdn_model.get_weights()[1]
        w2 = mdn_model.get_weights()[2]
        b2 = mdn_model.get_weights()[3]
            
        
    else:
        mdn_model = None

    dsmc = DSMCSimulation(n_particles=args.n_particles, n_steps=args.n_steps, time_step=args.time_step, mdn_model=mdn_model)
    dsmc.run_simulation()
    print(f"Total elastic collisions: {dsmc.elastic_collisions}")
    print(f"Total inelastic collisions: {dsmc.inelastic_collisions}")
    print(f"Total rejected collisions percentage: {dsmc.rejected_collisions/ (dsmc.elastic_collisions + dsmc.inelastic_collisions + dsmc.rejected_collisions) * 100}%")
    dsmc.plot_energy_relaxation_T()
    dsmc.plot_positions()