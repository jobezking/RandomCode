import matplotlib.pyplot as plt
import numpy as np

def plot_linear_equation(slope, intercept, x_range=(-10, 10)):
    """
    Plots a line given its slope and y-intercept.

    Args:
        slope (float): The slope (m) of the linear equation (y = mx + b).
        intercept (float): The y-intercept (b) of the linear equation (y = mx + b).
        x_range (tuple): A tuple (min_x, max_x) defining the range for x-values.
    """
    # Generate a range of x-values
    x = np.linspace(x_range[0], x_range[1], 100)  # 100 points for a smooth line

    # Calculate corresponding y-values using the linear equation
    y = slope * x + intercept

    # Plot the line
    plt.plot(x, y, label=f'y = {slope}x + {intercept}')

    # Add labels and title
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Plot of a Linear Equation')
    plt.grid(True)  # Add a grid for better readability
    plt.axhline(0, color='black', linewidth=0.5) # Add x-axis
    plt.axvline(0, color='black', linewidth=0.5) # Add y-axis
    plt.legend()
    plt.show()

# Example usage:
# Plot y = 2x + 3
plot_linear_equation(slope=2, intercept=3)

# Plot y = -0.5x + 1
plot_linear_equation(slope=-0.5, intercept=1, x_range=(-5, 5))
