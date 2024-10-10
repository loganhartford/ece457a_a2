import numpy as np
import matplotlib.pyplot as plt

def easom(x1, x2):
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)

# Chat gpt made this function
def plot_easom_function():
    x1_vals = np.arange(-100, 101, 0.1)
    x2_vals = np.arange(-100, 101, 0.1)
    x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

    z_vals = easom(x1_grid, x2_grid)

    plt.figure(figsize=(10, 8))
    plt.contourf(x1_grid, x2_grid, z_vals, levels=50, cmap='viridis')
    plt.colorbar(label='Easom Function Value')
    plt.title('Easom Function Plot')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def simulated_annealing(initial_solution, initial_temp, cooling_rate, max_iter, min_temp, step_size=5):
    x1_current, x2_current = initial_solution
    f_current = easom(x1_current, x2_current)
    x1_best, x2_best, f_best = x1_current, x2_current, f_current
    T = initial_temp

    f_history = [f_best]

    for i in range(max_iter):
        # Generate neighbour
        x1_new = x1_current + np.random.uniform(-step_size, step_size)
        x2_new = x2_current + np.random.uniform(-step_size, step_size)
        f_new = easom(x1_new, x2_new)

        diff = f_new - f_current
        metropolis = np.exp(-(diff) / T)
        prob = np.random.rand()
        # print(f"diff: {diff}, metropolis: {metropolis}, prob: {prob}")
        if diff < 0 or prob < metropolis:
            x1_current, x2_current, f_current = x1_new, x2_new, f_new

        # Update best solution
        if f_current < f_best:
            x1_best, x2_best, f_best = x1_current, x2_current, f_current

        f_history.append(f_best)

        # Cool down
        T *= cooling_rate

        if i % 100 == 0:
            print(f"Iteration {i}, Best solution: {f_best}")

    return (x1_best, x2_best), f_best, f_history
