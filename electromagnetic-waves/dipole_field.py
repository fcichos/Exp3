# %% Cell 1
import numpy as np
import matplotlib.pyplot as plt

# %% Cell 2
def electric_field(r_vec, t, omega, p_vec, epsilon_0=8.85e-12, c=3e8):
    """
    Calculate the electric field vector at position r_vec and time t

    Parameters:
    -----------
    r_vec : numpy.ndarray
        Position vector (x, y, z) in meters
    t : float
        Time in seconds
    omega : float
        Angular frequency in rad/s
    p_vec : numpy.ndarray
        Dipole moment vector (px, py, pz) in Coulomb-meters
    epsilon_0 : float
        Vacuum permittivity in F/m
    c : float
        Speed of light in m/s

    Returns:
    --------
    numpy.ndarray
        Electric field vector (Ex, Ey, Ez) in V/m
    """

    # Calculate wave number k
    k = omega / c

    # Calculate magnitude of position vector
    r = np.linalg.norm(r_vec)

    # Calculate unit vector in radial direction
    e_r = r_vec / r

    # First term: ((e_r × p) × e_r) / (kr)
    cross1 = np.cross(e_r, p_vec)
    cross2 = np.cross(cross1, e_r)
    term1 = cross2 / (k * r)

    # Second term: 3(e_r(e_r · p) - p)(1/(kr)^3 - i/(kr)^2)
    p_dot_er = np.dot(e_r, p_vec)
    er_dot_p = e_r * p_dot_er
    term2 = 3 * (er_dot_p - p_vec) * (1/(k*r)**3 - 1j/(k*r)**2)

    # Combine terms and multiply by prefactor
    prefactor = (omega**3)/(4 * np.pi * epsilon_0 * c**3)

    # Phase factor
    phase = np.exp(1j * (k*r - omega*t))

    # Final electric field vector
    E = prefactor * (term1 + term2) * phase

    return E


# %% Cell 3
r_vec = np.array([1.0, 0.0, 0.0])  # 1 meter along x-axis
t = 0.0  # time = 0
omega = 2 * np.pi * 1e9  # 1 GHz
p_vec = np.array([0.0, 0.0, 1e-12])  # dipole along z-axis

# Calculate electric field
E_field = electric_field(r_vec, t, omega, p_vec)

E_field
# %% Cell 4
def plot_dipole_field():
    # Parameters
    omega = 2 * np.pi * 1e9  # 1 GHz
    p_vec = np.array([0.0, 1e-12,0])  # dipole along z-axis
    t = 0.0  # time = 0

    # Create grid of points
    n_points = 21  # number of points in each dimension
    grid_size = 1e-6  # size of grid in meters
    x = np.linspace(-grid_size, grid_size, n_points)
    z = np.linspace(-grid_size, grid_size, n_points)
    X, Z = np.meshgrid(x, z)

    # Calculate E-field at each point
    Ex = np.zeros((n_points, n_points))
    Ez = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(n_points):
            # Define minimum distance to avoid singularity
            min_distance = grid_size/n_points

            # Skip points very close to origin
            if np.sqrt(X[i,j]**2 + Z[i,j]**2) < min_distance:
                Ex[i,j] = np.nan
                Ez[i,j] = np.nan
                continue

            r_vec = np.array([X[i,j], 0, Z[i,j]])
            E = electric_field(r_vec, t, omega, p_vec)
            Ex[i,j] = np.real(E[0])
            Ez[i,j] = np.real(E[2])

    # Calculate magnitude and handle normalization
    E_magnitude = np.sqrt(Ex**2 + Ez**2)
    max_magnitude = np.nanmax(E_magnitude[~np.isnan(E_magnitude)])

    # Normalize vectors, handling NaN values
    Ex_norm = np.zeros_like(Ex)
    Ez_norm = np.zeros_like(Ez)
    mask = ~np.isnan(E_magnitude)
    Ex_norm[mask] = Ex[mask] / max_magnitude
    Ez_norm[mask] = Ez[mask] / max_magnitude

    # Create the plot
    plt.figure(figsize=(10, 10))

    # Plot only where we have valid values
    plt.quiver(X[mask], Z[mask], Ex_norm[mask], Ez_norm[mask],
              E_magnitude[mask], pivot='middle', cmap='viridis')

    # Plot dipole
    plt.plot([0], [0], 'ro', markersize=10, label='Dipole')

    # Add labels and title
    plt.xlabel('x (meters)')
    plt.ylabel('z (meters)')
    plt.title('Electric Field of an Oscillating Dipole\nin x-z plane (y=0)')
    plt.colorbar(label='Field Strength (normalized)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


# %% Cell 5
#
#
plot_dipole_field()
