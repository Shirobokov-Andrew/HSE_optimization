import oracles
import optimization
import numpy as np
import scipy
import matplotlib.pyplot as plt


def run_optimizer(oracle, line_search, n):
    [x_star, msg, history] = optimization.newton(oracle=oracle,
                                                        x_0=np.zeros(n),
                                                        tolerance=1e-9,
                                                        max_iter=100000000,
                                                        line_search_options=line_search,
                                                        trace=True)
    return msg, history


np.random.seed(42)

m = 1000
n = 300
regcoeff = 1 / m
A = 5 * scipy.sparse.csr_matrix(np.random.randn(m, n))
b = np.random.choice(np.array([-1, 1]), size=m, replace=True)

cs = [0.2, 0.5, 0.7, 1, 1.7]
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
# plt.title('GD on LogReg oracle with Armijo line search', fontsize=20)
# plt.legend(fontsize=20)
# plt.grid()
# plt.show()


for i, c2 in enumerate(c2s):
    oracle = oracles.create_log_reg_oracle(A, b, regcoeff)
    msg, history = run_optimizer(oracle, {'method': 'Wolfe', 'c2': c2}, n)
    x_values = np.arange(1, len(history['func']) + 1)
    y_values = np.log10(np.array(history['grad_norm']) ** 2 / history['grad_norm'][0] ** 2)
    plt.plot(x_values, y_values, label=f'Wolfe, c1=1e-4, c2={c2}', linewidth=3)
    plt.xlabel('k, number of iterations', fontsize=20)
    plt.ylabel(r'$\frac{||grad_k f||^2}{||grad_0 f||^2}$', fontsize=20)
# plt.title('GD on LogReg oracle with Wolfe line search (c1=1e-4)', fontsize=20)
# plt.legend(fontsize=20)
# plt.grid()
# plt.show()


for i, c in enumerate(cs):
    oracle = oracles.create_log_reg_oracle(A, b, regcoeff)
    msg, history = run_optimizer(oracle, {'method': 'Constant', 'c': c}, n)
    x_values = np.arange(1, len(history['func']) + 1)
    y_values = np.log10(np.array(history['grad_norm']) ** 2 / history['grad_norm'][0] ** 2)
    plt.plot(x_values, y_values, label=f'Constant, c={c}', linewidth=3)
    plt.xlabel('k, number of iterations', fontsize=20)
    plt.ylabel(r'$\frac{||grad_k f||^2}{||grad_0 f||^2}$', fontsize=20)
plt.title('Newton on LogReg oracle with different line search strategies and its constants', fontsize=20)
plt.legend(fontsize=10)
plt.grid()
plt.show()
