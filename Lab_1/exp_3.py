from sklearn.datasets import load_svmlight_file
import oracles
import optimization
import numpy as np
import matplotlib.pyplot as plt


def get_data(path):
    data = load_svmlight_file(path)
    return data[0], data[1]


def run_optimizer(A, b, m, n, method, method_func, tolerance):
    oracle = oracles.create_log_reg_oracle(A, b, 1 / m)
    [x_star, msg, history] = method_func(oracle=oracle,
                                         x_0=np.zeros(n),
                                         max_iter=200000,
                                         tolerance=tolerance,
                                         line_search_options={'method': method},
                                         trace=True,
                                         )
    return [x_star, msg, history]


def plotter(history_gd, history_nw, title, axes):
    axes[0].plot(history_gd['time'], history_gd['func'], color='red', marker='<', label=f'GD')
    axes[0].plot(history_nw['time'], history_nw['func'], color='blue', marker='*', label=f'Newton')
    axes[0].set_xlabel('running time, s')
    axes[0].set_ylabel('logreg loss function')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(history_gd['time'], np.log10(np.array(history_gd['grad_norm']) ** 2 / history_gd['grad_norm'][0] ** 2), color='red', marker='<', label=f'GD')
    axes[1].plot(history_nw['time'], np.log10(np.array(history_nw['grad_norm']) ** 2 / history_nw['grad_norm'][0] ** 2), color='blue', marker='*', label=f'Newton')
    axes[1].set_xlabel('running time, s')
    axes[1].set_ylabel('log10(grad^2/grad0^2)')
    axes[1].set_title(title)
    axes[1].legend()
    axes[1].grid()


path = "./real-sim.bz2"
dataset_name = 'real-sim'
A, b = get_data(path)
print(A.count_nonzero())
m = b.shape[0]
n = A.shape[1]
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

[x_star_gd, msg_gd, history_gd] = run_optimizer(A, b, m, n, 'Armijo', optimization.gradient_descent, tolerance=1e-5)
[x_star_nw, msg_nw, history_nw] = run_optimizer(A, b, m, n, 'Armijo', optimization.newton, tolerance=1e-9)
plotter(history_gd, history_nw, 'Armijo (c1=1e-4)', [ax[0, 0], ax[0, 1]])

[x_star_gd, msg_gd, history_gd] = run_optimizer(A, b, m, n, 'Wolfe', optimization.gradient_descent, tolerance=1e-5)
[x_star_nw, msg_nw, history_nw] = run_optimizer(A, b, m, n, 'Wolfe', optimization.newton, tolerance=1e-9)
plotter(history_gd, history_nw, 'Wolfe (c1=1e-4, c2=0.9)', [ax[1, 0], ax[1, 1]])

[x_star_gd, msg_gd, history_gd] = run_optimizer(A, b, m, n, 'Constant', optimization.gradient_descent, tolerance=1e-5)
[x_star_nw, msg_nw, history_nw] = run_optimizer(A, b, m, n, 'Constant', optimization.newton, tolerance=1e-9)
plotter(history_gd, history_nw, 'Constant (c=1.0)', [ax[2, 0], ax[2, 1]])

fig.suptitle(f'GD and Newton methods for {dataset_name} dataset', fontsize=15)
plt.subplots_adjust(left=0.125, bottom=0.074, right=0.9, top=0.937, wspace=0.2, hspace=0.38)
plt.show()

