import oracles
import optimization
import numpy as np
import scipy
import matplotlib.pyplot as plt


def run_optimizer(oracle, line_search, n):
    [x_star, msg, history] = optimization.gradient_descent(oracle=oracle,
                                                           x_0=np.zeros(n),
                                                           tolerance=1e-9,
                                                           max_iter=100000000,
                                                           line_search_options=line_search,
                                                           trace=True)
    return msg, history


np.random.seed(42)

######################################################################################################################
######################################################################################################################
################################################## Quadratic Oracle ##################################################
######################################################################################################################
######################################################################################################################

n = 10
A = np.random.randint(low=1, high=10, size=(n, n)).astype(np.float64)
A = A.T @ A
b = np.random.randint(low=1, high=10, size=n).astype(np.float64)
f_min_an = - 1 / 2 * b @ (np.linalg.inv(A) @ b)

cs = [10e-5, 7e-5, 5e-5, 3e-5, 2e-5]
c1s = [1e-10, 1e-7, 1e-4, 1e-1, 0.49]
c2s = [0.3, 0.6, 0.9, 0.99]

for i, c1 in enumerate(c1s):
    oracle = oracles.QuadraticOracle(A, b)
    msg, history = run_optimizer(oracle, {'method': 'Armijo', 'c1': c1}, n)
    x_values = np.arange(1, len(history['func']) + 1)
    y_values = np.log10(np.abs(((np.array(history['func']) - f_min_an) / f_min_an)))
    plt.plot(x_values, y_values, label=f'Armijo, c1={c1}', linewidth=3)
    plt.xlabel('k, number of iterations', fontsize=20)
    plt.ylabel(r'$\log10\left(|\frac{f_k - f^*}{f^*}|\right)$', fontsize=20)


for i, c2 in enumerate(c2s):
    oracle = oracles.QuadraticOracle(A, b)
    msg, history = run_optimizer(oracle, {'method': 'Wolfe', 'c2': c2}, n)
    x_values = np.arange(1, len(history['func']) + 1)
    y_values = np.log10(np.abs((np.array(history['func']) - f_min_an) / f_min_an))
    plt.plot(x_values, y_values, label=f'Wolfe, c1=1e-4, c2={c2}', linewidth=3)
    plt.xlabel('k, number of iterations', fontsize=20)
    plt.ylabel(r'$\log10\left(|\frac{f_k - f^*}{f_*}|\right)$', fontsize=20)
plt.title('GD on Quadratic oracle with Armijo and Wolfe line searches', fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.show()


for i, c in enumerate(cs):
    oracle = oracles.QuadraticOracle(A, b)
    msg, history = run_optimizer(oracle, {'method': 'Constant', 'c': c}, n)
    x_values = np.arange(1, len(history['func']) + 1)
    y_values = np.log10(np.abs((np.array(history['func']) - f_min_an) / f_min_an))
    plt.plot(x_values, y_values, label=f'Constant, c={c}', linewidth=3)
    plt.xlabel('k, number of iterations', fontsize=20)
    plt.ylabel(r'$\log10\left(|\frac{f_k - f^*}{f_*}|\right)$', fontsize=20)
plt.title('GD on Quadratic oracle with Constant line search', fontsize=20)
plt.legend(fontsize=15)
plt.grid()
plt.show()

######################################################################################################################
######################################################################################################################
################################################## LogReg Oracle #####################################################
######################################################################################################################
######################################################################################################################
m = 1000
n = 300
regcoeff = 1 / m
A = 5 * scipy.sparse.csr_matrix(np.random.randn(m, n))
b = np.random.choice(np.array([-1, 1]), size=m, replace=True)

cs = [0.19, 1e-1, 5e-2, 2e-2]
c1s = [1e-10, 1e-7, 1e-4, 1e-1, 0.49]
c2s = [0.001, 0.1, 0.99]

for i, c1 in enumerate(c1s):
    oracle = oracles.create_log_reg_oracle(A, b, regcoeff)
    msg, history = run_optimizer(oracle, {'method': 'Armijo', 'c1': c1}, n)
    x_values = np.arange(1, len(history['func']) + 1)
    y_values = np.log10(np.array(history['grad_norm']) ** 2 / history['grad_norm'][0] ** 2)
    plt.plot(x_values, y_values, label=f'Armijo, c1={c1}', linewidth=3)
    plt.xlabel('k, number of iterations', fontsize=20)
    plt.ylabel(r'$\frac{||grad_k f||^2}{||grad_0 f||^2}$', fontsize=20)


for i, c2 in enumerate(c2s):
    oracle = oracles.create_log_reg_oracle(A, b, regcoeff)
    msg, history = run_optimizer(oracle, {'method': 'Wolfe', 'c2': c2}, n)
    x_values = np.arange(1, len(history['func']) + 1)
    y_values = np.log10(np.array(history['grad_norm']) ** 2 / history['grad_norm'][0] ** 2)
    plt.plot(x_values, y_values, label=f'Wolfe, c1=1e-4, c2={c2}', linewidth=3)
    plt.xlabel('k, number of iterations', fontsize=20)
    plt.ylabel(r'$\frac{||grad_k f||^2}{||grad_0 f||^2}$', fontsize=20)
plt.title('GD on LogReg oracle with Armijo and Wolfe line searches', fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.show()


for i, c in enumerate(cs):
    oracle = oracles.create_log_reg_oracle(A, b, regcoeff)
    msg, history = run_optimizer(oracle, {'method': 'Constant', 'c': c}, n)
    x_values = np.arange(1, len(history['func']) + 1)
    y_values = np.log10(np.array(history['grad_norm']) ** 2 / history['grad_norm'][0] ** 2)
    plt.plot(x_values, y_values, label=f'Constant, c={c}', linewidth=3)
    plt.xlabel('k, number of iterations', fontsize=20)
    plt.ylabel(r'$\frac{||grad_k f||^2}{||grad_0 f||^2}$', fontsize=20)
plt.title('GD on LogReg oracle with Constant line search', fontsize=20)
plt.legend(fontsize=17)
plt.grid()
plt.show()
