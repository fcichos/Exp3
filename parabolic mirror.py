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



```python
import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        # Initialize weights matrix with zeros
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """Train the network with a list of patterns"""
        # Clear previous weights
        self.weights = np.zeros((self.size, self.size))

        # For each pattern
        for pattern in patterns:
            # Convert to bipolar (-1, 1)
            pattern = np.array(pattern) * 2 - 1
            # Outer product of pattern with itself
            self.weights += np.outer(pattern, pattern)

        # Set diagonal to zero and divide by number of patterns
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def recall(self, pattern, max_iterations=10):
        """Recall a pattern from the network"""
        # Convert to bipolar
        state = np.array(pattern) * 2 - 1

        # Iterate until convergence or max iterations
        for _ in range(max_iterations):
            state_old = state.copy()

            # Update each neuron
            for i in range(self.size):
                # Calculate activation
                activation = np.dot(self.weights[i], state)
                # Apply threshold function
                state[i] = 1 if activation >= 0 else -1

            # Check if network has converged
            if np.array_equal(state, state_old):
                break

        # Convert back to binary
        return (state + 1) // 2

# Example usage
def main():
    # Create patterns (3x3 binary patterns)
    patterns = [
        [1,1,1,
         0,0,0,
         1,1,1],  # Pattern 1: horizontal lines

        [1,0,1,
         1,0,1,
         1,0,1]   # Pattern 2: vertical lines
    ]

    # Create network
    network = HopfieldNetwork(9)

    # Train network
    network.train(patterns)

    # Test with noisy pattern (corrupted version of pattern 1)
    test_pattern = [1,1,0,  # Last bit of first row flipped
                   0,0,0,
                   1,1,1]

    # Recall pattern
    result = network.recall(test_pattern)

    # Print results
    print("Test Pattern:")
    print_pattern(test_pattern)
    print("\nRecalled Pattern:")
    print_pattern(result)

def print_pattern(pattern):
    """Helper function to print 3x3 patterns"""
    for i in range(0, 9, 3):
        print(pattern[i:i+3])


main()
```
