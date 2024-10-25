# %% cell 1
import numpy as np
import matplotlib.pyplot as plt

def parabolic_mirror(x, a):
    return a * x**2

def tangent(x, a, k):
    return 2*a*k*(x-k) + a*k**2

def normal(x, a, k):
    return -1/(2*a*k)*(x-k) + a*k**2

def reflected_ray(x, a, k):
    # Slope of the normal line
    m_normal = -1/(2*a*k)

    # Angle of the normal with respect to the vertical (incident ray)
    theta_normal = np.arctan(m_normal)

    # Angle of reflection (same as angle of incidence)
    theta_reflection = theta_normal

    # Slope of the reflected ray
    m_reflected = np.tan(2 * theta_reflection)

    return m_reflected*(x - k) + a*k**2

# Set up the plot
x = np.linspace(-10, 10, 1000)
a = 0.1
k = 5

# Plot parabola
y = parabolic_mirror(x, a)
plt.plot(x, y, label='Parabola')

# Plot tangent line
t = tangent(x, a, k)
plt.plot(x, t, label='Tangent')

# Plot normal line
x_normal = np.linspace(k-2, k+2, 100)
m = normal(x_normal, a, k)
plt.plot(x_normal, m, "k--", label='Normal')

# Plot incident ray
plt.arrow(k, 10, 0, -5, color='r', width=0.05, head_width=0.3, head_length=0.3, label='Incident Ray')

# Plot reflected ray
x_reflected = np.linspace(k, -10, 100)
r = reflected_ray(x_reflected, a, k)
plt.plot(x_reflected, r, 'g-', label='Reflected Ray')

# Add focal point
f = 1/(4*a)
plt.plot(0, f, 'ro', label='Focal Point')

# Add point of reflection
plt.plot(k, parabolic_mirror(k, a), 'bo', label='Reflection Point')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Parabolic Mirror with Incident and Reflected Ray')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(-10, 10)
plt.ylim(0, 10)
plt.show()
