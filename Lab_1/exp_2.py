import numpy as np
import matplotlib.pyplot as plt
import oracles
import optimization


def plotter(x, y, title=None, legend=None):
    for i in range(y.shape[0]):
        plt.plot(x, y.mean(axis=1)[i, :])
    plt.xlabel('k, condition number', fontsize=15)
    plt.ylabel('T(n, k), avg number of iterations', fontsize=15)
    plt.title(title, fontsize=20)
    plt.grid()
    plt.legend(legend, fontsize=20)


def run_iters(method, avg_iter_numbers):
    for j, n in enumerate(dims):
        for k in range(inner_iter_number):
            A_matrices = np.array([np.diag(np.random.choice(np.arange(1, cond), replace=True, size=n)) for cond in cond_numbers])
            A_matrices[:, 0, 0] = 1
            A_matrices[:, n - 1, n - 1] = cond_numbers.copy()
            b_vectors = np.random.randn(A_matrices.shape[0], A_matrices.shape[1])
            iter_numbers = []
            for i in range(A_matrices.shape[0]):
                oracle = oracles.QuadraticOracle(A_matrices[i, :, :] / cond_numbers[i], b_vectors[i, :])
                [x_star, msg, history] = optimization.gradient_descent(oracle=oracle,
                                                                       x_0=np.zeros(n),
                                                                       tolerance=1e-9,
                                                                       max_iter=100000,
                                                                       line_search_options={'method': method},
                                                                       trace=True)
                iter_numbers.append(len(history['time']))
            avg_iter_numbers[j, k, :] = np.array([iter_numbers])
        print(f'dim={n}')
    return avg_iter_numbers


np.random.seed(42)
cond_numbers = np.arange(2, 1100, 100)
colors = ['blue', 'green', 'red', 'magenta']
markers = ['.', 'o', '<', '*']
dims = np.array([4, 16, 64, 256])

plt.figure()
inner_iter_number = 5
avg_iter_numbers = np.zeros((dims.shape[0], inner_iter_number, cond_numbers.shape[0]))
avg_iter_numbers = run_iters('Armijo', avg_iter_numbers)
plotter(cond_numbers, avg_iter_numbers, 'Avg number of iterations of GD with Armijo line search (c1=1e-4)', [f'n={n}' for n in dims])
plt.show()

plt.figure()
inner_iter_number = 15
avg_iter_numbers = np.zeros((dims.shape[0], inner_iter_number, cond_numbers.shape[0]))
avg_iter_numbers = run_iters('Wolfe', avg_iter_numbers)
plotter(cond_numbers, avg_iter_numbers, 'Avg number of iterations of GD with Wolfe line search (c1=1e-4, c2=0.9)', [f'n={n}' for n in dims])
plt.show()

plt.figure()
inner_iter_number = 10
avg_iter_numbers = np.zeros((dims.shape[0], inner_iter_number, cond_numbers.shape[0]))
avg_iter_numbers = run_iters('Constant', avg_iter_numbers)
plotter(cond_numbers, avg_iter_numbers, 'Avg number of iterations of GD with Constant line search (c=0.001)', [f'n={n}' for n in dims])
plt.show()
