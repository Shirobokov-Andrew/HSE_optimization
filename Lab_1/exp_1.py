import numpy as np
import matplotlib.pyplot as plt
import oracles
import optimization


def plot_levels(func, xrange=None, yrange=None, levels=None, ax=None):
    """
    Plotting the contour lines of the function.

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> plot_levels(oracle.func)
    """

    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    CS = ax.contour(X, Y, Z, levels=levels, colors='k', linewidth=4.0)
    ax.clabel(CS, inline=1, fontsize=8)
    ax.grid()


def plot_trajectory(history, label=None, ax=None):
    """
    Plotting the trajectory of a method.
    Use after plot_levels(...).

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> [x_star, msg, history] = optimization.gradient_descent(oracle, np.array([3.0, 1.5], trace=True)
    >> plot_levels(oracle.func)
    >> plot_trajectory(oracle.func, history['x'])
    """
    x_values, y_values = zip(*history)
    ax.plot(x_values, y_values, '-v', linewidth=1.0, ms=3.0, alpha=1.0, c='r', label=label)
    ax.set_xlabel(r'$x_1$', fontsize=15)
    ax.set_ylabel(r'$x_2$', fontsize=15)
    ax.set_title(label, fontsize=20)


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))
levels = np.hstack((np.arange(-25, 20, 5), np.geomspace(30, 600, 10)))
xrange = [2, 15]
yrange = [-11, 0]

A_base = np.array([[1, 2], [3, 4]])
A = A_base.T @ A_base
print(f'A matrix: \n {A}')
print(f'A matrix cond number: {np.linalg.cond(A)}')
b = np.array([5, 3])
x_0 = np.array([4, -6])
oracle = oracles.QuadraticOracle(A, b)
[x_star, msg, history] = optimization.gradient_descent(oracle=oracle,
                                                       x_0=x_0,
                                                       tolerance=1e-15,
                                                       max_iter=100000,
                                                       line_search_options={'method': 'Armijo'},
                                                       trace=True)
plot_levels(oracle.func, levels=levels, xrange=xrange, yrange=yrange, ax=ax[0])
plot_trajectory(history['x'], ax=ax[0], label='Armijo (c1=1e-4)')


oracle = oracles.QuadraticOracle(A, b)
[x_star, msg, history] = optimization.gradient_descent(oracle=oracle,
                                                       x_0=x_0,
                                                       tolerance=1e-15,
                                                       max_iter=100000,
                                                       line_search_options={'method': 'Wolfe', 'c2': 0.9},
                                                       trace=True)

plot_levels(oracle.func, levels=levels, xrange=xrange, yrange=yrange, ax=ax[1])
plot_trajectory(history['x'], ax=ax[1], label='Wolfe (c1=1e-4, c2=0.9)')

oracle = oracles.QuadraticOracle(A, b)
[x_star, msg, history] = optimization.gradient_descent(oracle=oracle,
                                                       x_0=x_0,
                                                       tolerance=1e-15,
                                                       max_iter=100000,
                                                       line_search_options={'method': 'Constant', 'c': 0.01},
                                                       trace=True)

plot_levels(oracle.func, levels=levels, xrange=xrange, yrange=yrange, ax=ax[2])
plot_trajectory(history['x'], ax=ax[2], label='Constant (c=0.01)')

plt.subplots_adjust(left=0.048, bottom=0.223, right=0.987, top=0.815, wspace=0.149, hspace=0.272)
fig.suptitle('Gradient descent for matrix A1', fontsize=25)
plt.show()



fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
levels = np.hstack((np.arange(-3, 21, 3), np.geomspace(21, 300, 7)))
xrange = [-3, 5]
yrange = [-2.5, 3.2]

A_base = np.array([[1.2, 0], [0, 1]])
A = A_base.T @ A_base
print(f'A matrix: \n {A}')
print(f'A matrix cond number: {np.linalg.cond(A)}')
b = np.array([2, 3])
x_0 = np.array([4.5, -2])
oracle = oracles.QuadraticOracle(A, b)
[x_star, msg, history] = optimization.gradient_descent(oracle=oracle,
                                                       x_0=x_0,
                                                       tolerance=1e-15,
                                                       max_iter=100000,
                                                       line_search_options={'method': 'Armijo'},
                                                       trace=True)
plot_levels(oracle.func, levels=levels, xrange=xrange, yrange=yrange, ax=ax[0])
plot_trajectory(history['x'], ax=ax[0], label='Armijo (c1=1e-4))')

oracle = oracles.QuadraticOracle(A, b)
[x_star, msg, history] = optimization.gradient_descent(oracle=oracle,
                                                       x_0=x_0,
                                                       tolerance=1e-15,
                                                       max_iter=100000,
                                                       line_search_options={'method': 'Wolfe'},
                                                       trace=True)

plot_levels(oracle.func, levels=levels, xrange=xrange, yrange=yrange, ax=ax[1])
plot_trajectory(history['x'], ax=ax[1], label='Wolfe (c1=1e-4, c2=0.9)')


oracle = oracles.QuadraticOracle(A, b)
[x_star, msg, history] = optimization.gradient_descent(oracle=oracle,
                                                       x_0=x_0,
                                                       tolerance=1e-15,
                                                       max_iter=100000,
                                                       line_search_options={'method': 'Constant', 'c': 0.01},
                                                       trace=True)

plot_levels(oracle.func, levels=levels, xrange=xrange, yrange=yrange, ax=ax[2])
plot_trajectory(history['x'], ax=ax[2], label='Constant (c=0.01)')

plt.subplots_adjust(left=0.048, bottom=0.223, right=0.987, top=0.815, wspace=0.149, hspace=0.272)
fig.suptitle('Gradient descent for matrix A2', fontsize=25)
plt.show()
