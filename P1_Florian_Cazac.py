import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import argparse

# Function to maximize
# The negative sign is used because the optimizer minimizes the function
# but we want to maximize the profit
def profit(xy):
    x, y = xy
    return -( -x**2 + 4*x*y - 2 * y ** 2)


# Constraints
def constraint1(xy):
    x, y = xy
    return 30 - (x + 2*y)

def constraint2(xy):
    x, y = xy
    return x*y - 50

def constraint3(xy):
    x, y = xy
    return (3 * x**2) / 100 + 5 - y 


# Run the optimization using SLSQP
def optimize_problem(bounds):

    # Initial guess
    initial_guess = [5, 5]

    # Define the constraints for the optimizer
    constraints = [
        {'type': 'ineq', 'fun': constraint1},  # x + 2y <= 30
        {'type': 'ineq', 'fun': constraint2},  # xy >= 50
        {'type': 'ineq', 'fun': constraint3},  # y <= (3x^2)/100 + 5
    ]

    # Run the optimization
    result = minimize(profit, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x, -result.fun  # return optimal x, y, and the maximum profit


def parse_args():
    # Create the parser
    # The description and epilog arguments provide help text when the user runs the script with the --help flag
    parser = argparse.ArgumentParser(
            description="SLSQP Optimization for Resource Allocation",
            epilog=
            "Use the following format: python your_script.py --xmin <value> --xmax <value> --ymin <value> --ymax <value>"
        )    
    
    # Add the arguments
    parser.add_argument("--xmin", type=float, default=0, help="Minimum bound for x")
    parser.add_argument("--xmax", type=float, default=30, help="Maximum bound for x")
    parser.add_argument("--ymin", type=float, default=0, help="Minimum bound for y")
    parser.add_argument("--ymax", type=float, default=30, help="Maximum bound for y")

    return parser.parse_args()


def plot(bounds):

    # Run the optimization to get the optimal values and maximum profit
    optimal_values, max_profit = optimize_problem(bounds)

    # Create grid for contour plot
    x = np.linspace(bounds[0][0], bounds[0][1], 400)
    y = np.linspace(bounds[1][0], bounds[1][1], 400)
    X, Y = np.meshgrid(x, y)
    
    # Compute the objective function values for the grid
    Z = -X**2 + 4*X*Y - 2*Y**2

    # Feasible region constraints
    y1 = (30 - x) / 2  
    y2 = 50 / x        
    y3 = (3 * x**2) / 100 + 5  

    # Plot the constraints
    plt.figure(figsize=(10, 8))
    plt.plot(x, y1, label=r'$x + 2y \leq 30$', color='blue')
    plt.plot(x, y2, label=r'$xy \geq 50$', color='green')
    plt.plot(x, y3, label=r'$y \leq \frac{3x^2}{100} + 5$', color='red')

    # Fill feasible region
    plt.fill_between(x, np.maximum(y2, np.full_like(x, -np.inf)), np.minimum(y1, y3), where=(y2 <= y3) & (y2 <= y1), color='gray', alpha=0.5)

    # Plot the objective function
    contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

    # Plot the optimal point
    plt.plot(optimal_values[0], optimal_values[1], 'ro', label='Optimal Point')

    plt.xlim(0, 30)
    plt.ylim(0, 20)

    # Add labels and title
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Objective Function and Feasible Region')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def main():
    # Parse the command-line arguments
    args = parse_args()

    # Set bounds for x and y
    bounds = [(args.xmin, args.xmax), (args.ymin, args.ymax)]

    # Plot the feasible region
    plot(bounds)

    # Run the optimization
    optimal_values, max_profit = optimize_problem(bounds)

    # Output the results
    print("Optimization Results:\n")
    print(f"Optimal x: {optimal_values[0]:.4f}")
    print(f"Optimal y: {optimal_values[1]:.4f}")
    print(f"Maximum Profit: {max_profit:.4f}")

if __name__ == "__main__":
    main()