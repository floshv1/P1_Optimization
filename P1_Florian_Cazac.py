import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import argparse

def profit(xy):
    x, y = xy
    return -( -x**2 + 4*x*y - 2 * y ** 2)

def constraint1(xy):
    x, y = xy
    return x + 2*y - 30

def constraint2(xy):
    x, y = xy
    return x*y - 50

def constraint3(xy):
    x, y = xy
    return ((3*x **2)/100) + 5 - y


def parse_args():
    parser = argparse.ArgumentParser(description="SLSQP Optimization for Resource Allocation")
    parser.add_argument("--xmin", type=float, default=0, help="Minimum bound for x")
    parser.add_argument("--xmax", type=float, default=30, help="Maximum bound for x")
    parser.add_argument("--ymin", type=float, default=0, help="Minimum bound for y")
    parser.add_argument("--ymax", type=float, default=30, help="Maximum bound for y")
    return parser.parse_args()


def plot_feasible_region():
    x = np.linspace(1, 30, 400)
    y1 = (30 - x) / 2  # x + 2y <= 30
    y2 = 50 / x        # xy >= 50
    y3 = (3 * x**2) / 100 + 5  # y <= (3x^2)/100 + 5

    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, label=r'$x + 2y \leq 30$', color='blue')
    plt.plot(x, y2, label=r'$xy \geq 50$', color='green')
    plt.plot(x, y3, label=r'$y \leq \frac{3x^2}{100} + 5$', color='red')
    plt.fill_between(x, np.maximum(y2, np.full_like(x, -np.inf)), np.minimum(y1, y3), where=(y2 <= y3) & (y2 <= y1), color='gray', alpha=0.5)
    plt.xlim(0, 30)
    plt.ylim(0, 20)
    plt.xlabel('x (Resource allocated to Project 1)')
    plt.ylabel('y (Resource allocated to Project 2)')
    plt.title('Feasible Region for Resource Allocation Problem')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    # Set bounds for x and y
    bounds = [(args.xmin, args.xmax), (args.ymin, args.ymax)]

    # Plot the feasible region
    plot_feasible_region()
