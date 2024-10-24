import numpy as np
import matplotlib.pyplot as plt

def colider(v, crmax, z_eff, selxtra, coeff, sD, rot_energy, omega, mass):
    """
    Colider function to process collisions in cells.

    Inputs:
        v: Velocities of the particles
        crmax: Estimated maximum relative speed in a cell
        z_eff: Effective collision number
        selxtra: Extra selections carried over from last timestep
        coeff: Coefficient in computing number of selected pairs
        sD: Structure containing sorting lists
        rot_energy: Rotational energies of particles
        omega: Parameter for rotational energy distribution
        mass: Particle mass
    Outputs:
        v: Updated velocities of the particles
        crmax: Updated maximum relative speed
        selxtra: Extra selections carried over to next timestep
        col: Total number of collisions processed
    """

    ncell = sD['ncell']
    col = 0  # Count number of collisions

    # Loop over cells, processing collisions in each cell
    for jcell in range(ncell):

        # Skip cells with only one particle
        number = sD['cell_n'][jcell]
        if number > 1:

            # Determine number of candidate collision pairs
            # to be selected in this cell
            select = coeff * number * (number - 1) * crmax[jcell] + selxtra[jcell]
            nsel = int(np.floor(select))  # Number of pairs to be selected
            selxtra[jcell] = select - nsel  # Carry over any left-over fraction
            crm = crmax[jcell]  # Current maximum relative speed

            # Loop over total number of candidate collision pairs
            for isel in range(nsel):

                # Pick two particles at random out of this cell
                k = int(np.floor(np.random.rand() * number))
                kk = int(np.mod(np.ceil(k + np.random.rand() * (number - 1)), number))
                ip1 = sD['Xref'][k + sD['index'][jcell]]
                ip2 = sD['Xref'][kk + sD['index'][jcell]]

                # Calculate pair's relative speed
                cr = np.linalg.norm(v[ip1, :] - v[ip2, :])  # Relative speed
                if cr > crm:
                    crm = cr  # Update crm to larger value

                R1 = np.random.rand()
                m_r = mass / 2

                if R1 < 1 / z_eff:
                    # Elastic collision
                    col += 1  # Collision counter
                    vcm = 0.5 * (v[ip1, :] + v[ip2, :])  # Center of mass velocity
                    cos_th = 1 - 2 * np.random.rand()  # Cosine of collision angle
                    sin_th = np.sqrt(1 - cos_th**2)  # Sine of collision angle
                    phi = 2 * np.pi * np.random.rand()  # Collision angle phi
                    vrel = np.zeros(3)
                    vrel[0] = cr * cos_th
                    vrel[1] = cr * sin_th * np.cos(phi)
                    vrel[2] = cr * sin_th * np.sin(phi)
                    v[ip1, :] = vcm + 0.5 * vrel
                    v[ip2, :] = vcm - 0.5 * vrel
                else:
                    # Inelastic collision
                    col += 1  # Collision counter
                    E_tr = 0.5 * m_r * cr**2
                    E_i = rot_energy[ip1] + rot_energy[ip2]
                    E_tot = E_tr + E_i
                    loop = True
                    while loop:
                        R2 = np.random.rand()
                        a = (((5 / 2 - omega) / (3 / 2 - omega)) * R2) ** (3 / 2 - omega)
                        b = (3 / 2 - omega) * (1 - R2)
                        ratio = a * b
                        R3 = np.random.rand()
                        if ratio > R3:
                            loop = False
                    E_tr_post = R2 * E_tot
                    E_i_post = E_tot - E_tr_post
                    R_i = np.random.rand()
                    rot_energy[ip1] = R_i * E_i_post
                    rot_energy[ip2] = E_i_post - rot_energy[ip1]
                    vr = np.sqrt(2 * (E_tr_post / m_r))
                    vcm = 0.5 * (v[ip1, :] + v[ip2, :])
                    cos_th = 1 - 2 * np.random.rand()
                    sin_th = np.sqrt(1 - cos_th**2)
                    phi = 2 * np.pi * np.random.rand()
                    vrel = np.zeros(3)
                    vrel[0] = vr * cos_th
                    vrel[1] = vr * sin_th * np.cos(phi)
                    vrel[2] = vr * sin_th * np.sin(phi)
                    v[ip1, :] = vcm + 0.5 * vrel
                    v[ip2, :] = vcm - 0.5 * vrel

            crmax[jcell] = crm  # Update max relative speed

    return v, crmax, selxtra, col

def mover(x, v, npart, L, tau):
    """
    Mover function to move particles by free flight with periodic boundaries.

    Inputs:
        x: Positions of the particles
        v: Velocities of the particles
        npart: Number of particles
        L: System size
        tau: Time step
    Outputs:
        x, v: Updated positions and velocities
    """

    # Move all particles
    x = x + v[:, 0] * tau

    # Apply periodic boundary conditions
    x = np.mod(x, L)

    return x, v

def sampler(x, v, npart, L, sampD, rot_energy):
    """
    Sampler function to sample density, velocity and temperature

    Inputs:
        x: Particle positions
        v: Particle velocities
        npart: Number of particles
        L: System size
        sampD: Structure with sampling data
        rot_energy: Rotational energies of particles
    Outputs:
        sampD: Updated structure with sampling data
    """

    # Compute cell location for each particle
    ncell = sampD['ncell']
    jx = np.ceil(ncell * x / L).astype(int) - 1
    jx = np.minimum(jx, ncell - 1)

    # Initialize running sums of number, velocity and v^2
    sum_n = np.zeros(ncell)
    sum_v = np.zeros((ncell, 3))
    sum_v2 = np.zeros(ncell)
    sum_rot = np.zeros(ncell)

    # For each particle, accumulate running sums for its cell
    for ipart in range(npart):
        jcell = jx[ipart]  # Particle ipart is in cell jcell
        sum_n[jcell] += 1
        sum_v[jcell, :] += v[ipart, :]
        sum_v2[jcell] += v[ipart, 0] ** 2 + v[ipart, 1] ** 2 + v[ipart, 2] ** 2
        sum_rot[jcell] += rot_energy[ipart]

    # Avoid division by zero
    nonzero_cells = sum_n > 0
    sum_v[nonzero_cells, :] = sum_v[nonzero_cells, :] / sum_n[nonzero_cells, np.newaxis]
    sum_v2[nonzero_cells] = sum_v2[nonzero_cells] / sum_n[nonzero_cells]

    # Use current sums to update sample number, velocity and temperature
    sampD['ave_n'] += sum_n
    sampD['ave_u'] += sum_v
    temp_T = np.zeros(ncell)
    temp_T[nonzero_cells] = sum_v2[nonzero_cells] - (sum_v[nonzero_cells, 0] ** 2 +
                                                     sum_v[nonzero_cells, 1] ** 2 +
                                                     sum_v[nonzero_cells, 2] ** 2)
    sampD['ave_T'] += temp_T
    sampD['ave_rot'] += sum_rot
    sampD['nsamp'] += 1

    return sampD

def sorter(x, L, sD):
    """
    Sorter function to sort particles into cells

    Inputs:
        x: Positions of particles
        L: System size
        sD: Structure containing sorting lists
    Output:
        sD: Updated structure containing sorting lists
    """

    # Find the cell address for each particle
    npart = sD['npart']
    ncell = sD['ncell']
    jx = np.floor(x * ncell / L).astype(int)
    jx = np.minimum(jx, ncell - 1)

    # Count the number of particles in each cell
    sD['cell_n'] = np.zeros(ncell, dtype=int)
    for ipart in range(npart):
        sD['cell_n'][jx[ipart]] += 1

    # Build index list as cumulative sum of the number of particles in each cell
    sD['index'] = np.zeros(ncell, dtype=int)
    m = 0
    for jcell in range(ncell):
        sD['index'][jcell] = m
        m += sD['cell_n'][jcell]

    # Build cross-reference list
    temp = np.zeros(ncell, dtype=int)  # Temporary array
    for ipart in range(npart):
        jcell = jx[ipart]
        k = sD['index'][jcell] + temp[jcell]
        sD['Xref'][k] = ipart
        temp[jcell] += 1

    return sD

# Main code
import numpy as np
import matplotlib.pyplot as plt

# Initialize constants
boltz = 1.3806e-23  # Boltzmann's constant (J/K)
mass = 2.3258671e-26  # Mass of particle (kg)
diam = 4.14e-10  # Effective diameter of particle (m)
T = 273  # Initial temperature (K)
int_dof = 2  # Internal degrees of freedom
omega = 0.5  # 0.5 for HS, 0.74 for N2(VHS)

density = 2.387656008088409e+26
L = 25e-9  # System length
Volume = L * (10e-8)**3  # Volume

npart = int(input('Enter number of simulation particles: '))
eff_num = density * Volume / npart
print(f'Each simulation particle represents {eff_num} atoms')

mfp = Volume / (np.sqrt(2) * np.pi * diam**2 * npart * eff_num)
print(f'Knudsen number is {mfp/L}')

mpv = np.sqrt(2 * boltz * T / mass)  # Most probable initial velocity

# Assign random positions and velocities to particles
np.random.seed(1)
x = L * np.random.rand(npart)
# Assign thermal velocities using Gaussian random numbers
v = np.sqrt(boltz * T / mass) * np.random.randn(npart, 3)
# Assign random rotational energy
rot_energy = -np.log(np.random.rand(npart)) * int_dof * boltz * T / 2

z_eff = 6.375  # z_eff value

# Initialize variables used for evaluating collisions
ncell = 20  # Number of cells
tau = 0.2 * (L / ncell) / mpv  # Set timestep tau
vrmax = 3 * mpv * np.ones(ncell)  # Estimated max rel. speed in a cell
selxtra = np.zeros(ncell)  # Used by collision routine "colider"
coeff = 0.5 * eff_num * np.pi * diam**2 * tau / (Volume / ncell)

# Declare structure for lists used in sorting
sortData = {'ncell': ncell,
            'npart': npart,
            'cell_n': np.zeros(ncell, dtype=int),
            'index': np.zeros(ncell, dtype=int),
            'Xref': np.zeros(npart, dtype=int)}

# Initialize structure and variables used in statistical sampling
sampData = {'ncell': ncell,
            'nsamp': 0,
            'ave_n': np.zeros(ncell),
            'ave_u': np.zeros((ncell, 3)),
            'ave_rot': np.zeros(ncell),
            'ave_T': np.zeros(ncell)}
tsamp = 0  # Total sampling time

# Loop for the desired number of time steps
colSum = 0
nstep = int(input('Enter total number of timesteps: '))
for istep in range(nstep):

    # Move all the particles
    x, v = mover(x, v, npart, L, tau)

    # Sort the particles into cells
    sortData = sorter(x, L, sortData)

    # Evaluate collisions among the particles
    v, vrmax, selxtra, col = colider(v, vrmax, z_eff, selxtra, coeff, sortData, rot_energy, omega, mass)
    colSum += col

    # After initial transient, accumulate statistical samples
    if istep > nstep / 10:
        sampData = sampler(x, v, npart, L, sampData, rot_energy)
        tsamp += tau

    # Periodically display the current progress
    if istep % 10 == 0:
        print(f'Finished {istep} of {nstep} steps, Collisions = {colSum}')

# Normalize the accumulated statistics
nsamp = sampData['nsamp']
ave_n = ((eff_num / (Volume / ncell)) * sampData['ave_n'] / nsamp) / density
ave_u = sampData['ave_u'] / nsamp
ave_T_trans = (mass / (3 * boltz) * (sampData['ave_T'] / nsamp))
ave_T_rot = (1 / boltz) * (sampData['ave_rot'] / nsamp)
ave_T = (3 * ave_T_trans + 2 * ave_T_rot) / 5

xcell = (((np.arange(1, ncell + 1)) - 0.5) / ncell) * L

plt.figure()
plt.plot(xcell, ave_T)
plt.xlabel('position(x)')
plt.ylabel('Temperature')
plt.show()
