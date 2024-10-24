import numpy as np
import matplotlib.pyplot as plt

class DSMCSimulation:
    def __init__(self, npart, nstep):
        # Initialize constants
        self.boltz = 1.3806e-23    # Boltzmann's constant (J/K)
        self.mass = 2.3258671e-26  # Mass of particle (kg)
        self.diam = 4.14e-10       # Effective diameter of particle (m)
        self.T = 273               # Initial temperature (K)
        self.int_dof = 2           # Internal degrees of freedom
        self.omega = 0.5           # Collision model parameter

        # Initialize system parameters
        self.density = 2.387656008088409e+26  # Number density (m^-3)
        self.L = 25e-9                        # System length in x-direction (m)
        self.Ly = 1e-7                        # System length in y-direction (m)
        self.Lz = 1e-7                        # System length in z-direction (m)
        self.Volume = self.L * self.Ly * self.Lz  # System volume (m^3)

        self.npart = npart       # Number of simulation particles
        self.nstep = nstep       # Number of time steps

        # Effective number of real particles represented by one simulation particle
        self.eff_num = self.density * self.Volume / self.npart
        print(f'Each simulation particle represents {self.eff_num} atoms')

        # Mean free path and Knudsen number
        self.mfp = 1 / (np.sqrt(2) * np.pi * self.diam**2 * self.density)
        print(f'Knudsen number is {self.mfp / self.L}')

        # Most probable velocity
        self.mpv = np.sqrt(2 * self.boltz * self.T / self.mass)

        # Initialize random seed for reproducibility
        np.random.seed(1)

        # Assign random positions and velocities to particles
        self.x = self.L * np.random.rand(self.npart)  # Positions in x
        self.v = np.sqrt(self.boltz * self.T / self.mass) * np.random.randn(self.npart, 3)  # Velocities

        # Assign random rotational energy
        self.rot_energy = -np.log(np.random.rand(self.npart)) * self.int_dof * self.boltz * self.T / 2

        self.z_eff = 6.375  # Effective collision number

        # Initialize variables used for evaluating collisions
        self.ncell = 20  # Number of cells
        self.tau = 0.2 * (self.L / self.ncell) / self.mpv  # Time step
        self.vrmax = 3 * self.mpv * np.ones(self.ncell)  # Estimated max relative speed in a cell
        self.selxtra = np.zeros(self.ncell)  # Extra selections for collision routine
        self.coeff = 0.5 * self.eff_num * np.pi * self.diam**2 * self.tau / (self.Volume / self.ncell)

        # Declare structure for lists used in sorting
        self.sortData = {
            'ncell': self.ncell,
            'npart': self.npart,
            'cell_n': np.zeros(self.ncell, dtype=int),
            'index': np.zeros(self.ncell, dtype=int),
            'Xref': np.zeros(self.npart, dtype=int)
        }

        # Initialize structure and variables used in statistical sampling
        self.sampData = {
            'ncell': self.ncell,
            'nsamp': 0,
            'ave_n': np.zeros(self.ncell),
            'ave_u': np.zeros((self.ncell, 3)),
            'ave_rot': np.zeros(self.ncell),
            'ave_T': np.zeros(self.ncell)
        }
        self.tsamp = 0  # Total sampling time

    def mover(self):
        """Move all particles and apply periodic boundary conditions."""
        # Move all particles
        self.x += self.v[:, 0] * self.tau  # Update positions

        # Apply periodic boundary conditions
        self.x = self.x % self.L

    def sorter(self):
        """Sort particles into cells."""
        # Find the cell address for each particle
        jx = np.floor(self.x * self.ncell / self.L).astype(int)
        jx = np.minimum(jx, self.ncell - 1)

        # Count the number of particles in each cell
        self.sortData['cell_n'] = np.bincount(jx, minlength=self.ncell)

        # Build index list as cumulative sum of the number of particles in each cell
        self.sortData['index'][0] = 0
        for jcell in range(1, self.ncell):
            self.sortData['index'][jcell] = self.sortData['index'][jcell - 1] + self.sortData['cell_n'][jcell - 1]

        # Build cross-reference list
        temp = np.zeros(self.ncell, dtype=int)
        for ipart in range(self.npart):
            jcell = jx[ipart]
            k = self.sortData['index'][jcell] + temp[jcell]
            self.sortData['Xref'][k] = ipart
            temp[jcell] += 1

    def colider(self):
        """Process collisions in cells."""
        col = 0  # Collision counter
        # Loop over cells
        for jcell in range(self.ncell):
            number = self.sortData['cell_n'][jcell]
            if number > 1:
                # Determine number of candidate collision pairs
                select = self.coeff * number * (number - 1) * self.vrmax[jcell] + self.selxtra[jcell]
                nsel = int(np.floor(select))
                self.selxtra[jcell] = select - nsel
                crm = self.vrmax[jcell]
                m_r = self.mass / 2
                # Loop over total number of candidate collision pairs
                for isel in range(nsel):
                    # Pick two particles at random out of this cell
                    k = int(np.floor(np.random.rand() * number))
                    kk = int(np.mod(np.ceil(k + np.random.rand() * (number - 1)), number))
                    ip1 = self.sortData['Xref'][k + self.sortData['index'][jcell]]
                    ip2 = self.sortData['Xref'][kk + self.sortData['index'][jcell]]
                    # Calculate pair's relative speed
                    cr_vec = self.v[ip1] - self.v[ip2]
                    cr = np.linalg.norm(cr_vec)
                    if cr > crm:
                        crm = cr
                    # Decide whether to perform elastic or inelastic collision
                    if np.random.rand() < 1 / self.z_eff:
                        # Elastic collision
                        if cr / self.vrmax[jcell] > np.random.rand():
                            col += 1
                            vcm = 0.5 * (self.v[ip1] + self.v[ip2])
                            cos_th = 1 - 2 * np.random.rand()
                            sin_th = np.sqrt(1 - cos_th**2)
                            phi = 2 * np.pi * np.random.rand()
                            vrel = np.zeros(3)
                            vrel[0] = cr * cos_th
                            vrel[1] = cr * sin_th * np.cos(phi)
                            vrel[2] = cr * sin_th * np.sin(phi)
                            self.v[ip1] = vcm + 0.5 * vrel
                            self.v[ip2] = vcm - 0.5 * vrel
                    else:
                        # Inelastic collision
                        col += 1
                        E_tr = 0.5 * m_r * cr**2
                        E_i = self.rot_energy[ip1] + self.rot_energy[ip2]
                        E_tot = E_tr + E_i
                        # Energy redistribution loop
                        while True:
                            R2 = np.random.rand()
                            a = (((5 / 2 - self.omega) / (3 / 2 - self.omega)) * R2)**(3 / 2 - self.omega)
                            b = (3 / 2 - self.omega) * (1 - R2)
                            ratio = a * b
                            R3 = np.random.rand()
                            if ratio > R3:
                                break
                        E_tr_post = R2 * E_tot
                        E_i_post = E_tot - E_tr_post
                        R_i = np.random.rand()
                        self.rot_energy[ip1] = R_i * E_i_post
                        self.rot_energy[ip2] = E_i_post - self.rot_energy[ip1]
                        vr = np.sqrt(2 * (E_tr_post / m_r))
                        vcm = 0.5 * (self.v[ip1] + self.v[ip2])
                        cos_th = 1 - 2 * np.random.rand()
                        sin_th = np.sqrt(1 - cos_th**2)
                        phi = 2 * np.pi * np.random.rand()
                        vrel = np.zeros(3)
                        vrel[0] = vr * cos_th
                        vrel[1] = vr * sin_th * np.cos(phi)
                        vrel[2] = vr * sin_th * np.sin(phi)
                        self.v[ip1] = vcm + 0.5 * vrel
                        self.v[ip2] = vcm - 0.5 * vrel
                # Update crmax
                self.vrmax[jcell] = crm
        return col

    def sampler(self):
        """Sample density, velocity, and temperature."""
        # Compute cell location for each particle
        jx = np.floor(self.x * self.ncell / self.L).astype(int)
        jx = np.minimum(jx, self.ncell - 1)

        # Initialize running sums
        sum_n = np.bincount(jx, minlength=self.ncell)
        sum_v = np.zeros((self.ncell, 3))
        for i in range(3):
            sum_v[:, i] = np.bincount(jx, weights=self.v[:, i], minlength=self.ncell)
        sum_v2 = np.bincount(jx, weights=np.sum(self.v**2, axis=1), minlength=self.ncell)
        sum_rot = np.bincount(jx, weights=self.rot_energy, minlength=self.ncell)

        # Avoid division by zero
        sum_n_nonzero = sum_n.copy()
        sum_n_nonzero[sum_n_nonzero == 0] = 1  # To avoid division by zero

        # Update sample number, velocity, and temperature
        ave_v = sum_v / sum_n_nonzero[:, np.newaxis]
        ave_v2 = sum_v2 / sum_n_nonzero
        self.sampData['ave_n'] += sum_n
        self.sampData['ave_u'] += ave_v
        self.sampData['ave_T'] += ave_v2 - np.sum(ave_v**2, axis=1)
        self.sampData['ave_rot'] += sum_rot
        self.sampData['nsamp'] += 1

    def run_simulation(self):
        """Main simulation loop."""
        colSum = 0
        nstep_10 = self.nstep // 10
        for istep in range(1, self.nstep + 1):
            self.mover()
            self.sorter()
            col = self.colider()
            colSum += col
            if istep > nstep_10:
                self.sampler()
                self.tsamp += self.tau
            if istep % 10 == 0:
                print(f'Finished {istep} of {self.nstep} steps, Collisions = {colSum}')

    def compute_results(self):
        """Compute the averaged results after simulation."""
        nsamp = self.sampData['nsamp']
        ave_n = ((self.eff_num / (self.Volume / self.ncell)) * self.sampData['ave_n'] / nsamp) / self.density
        ave_u = self.sampData['ave_u'] / nsamp
        ave_T_trans = (self.mass / (3 * self.boltz)) * (self.sampData['ave_T'] / nsamp)
        ave_T_rot = (1 / self.boltz) * (self.sampData['ave_rot'] / nsamp)
        ave_T = (3 * ave_T_trans + 2 * ave_T_rot) / 5
        self.results = {
            'ave_n': ave_n,
            'ave_u': ave_u,
            'ave_T_trans': ave_T_trans,
            'ave_T_rot': ave_T_rot,
            'ave_T': ave_T
        }

    def plot_results(self):
        """Plot the results of the simulation."""
        xcell = ((np.arange(1, self.ncell + 1) - 0.5) / self.ncell) * self.L
        ave_T = self.results['ave_T']
        plt.figure(figsize=(10, 6))
        plt.plot(xcell, ave_T)
        plt.xlabel('Position (x)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Profile with Periodic Boundary Conditions')
        plt.grid(True)
        plt.show()

# Example usage:
if __name__ == "__main__":
    npart = int(input('Enter number of simulation particles: '))
    nstep = int(input('Enter total number of timesteps: '))

    dsmc = DSMCSimulation(npart=npart, nstep=nstep)
    dsmc.run_simulation()
    dsmc.compute_results()
    dsmc.plot_results()
