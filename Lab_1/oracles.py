import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if isinstance(A, scipy.sparse._csc.csc_matrix):
            if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A.data, A.T.data):
                raise ValueError('A should be a symmetric matrix.')
        elif not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        m = self.b.shape[0]
        m_vec = np.ones(m, dtype=np.float32) * 1 / m
        log_vec = np.logaddexp(0, -self.b * self.matvec_Ax(x))
        loss = m_vec @ log_vec + self.regcoef / 2 * np.linalg.norm(x) ** 2
        return loss

    def grad(self, x):
        m = self.b.shape[0]
        logit_vec = self.b * expit(-self.b * self.matvec_Ax(x))
        loss_grad = -1 / m * self.matvec_ATx(logit_vec) + self.regcoef * x
        return loss_grad

    def hess(self, x):
        m = self.b.shape[0]
        logits_matrix = scipy.sparse.diags(expit(self.b * self.matvec_Ax(x)) * (1 - expit(self.b * self.matvec_Ax(x))))
        hess_term_1 = 1 / m * self.matmat_ATsA(logits_matrix)
        n = hess_term_1.shape[0]
        hess_term_2 = self.regcoef * scipy.sparse.diags(np.ones(n))
        hess = hess_term_1 + hess_term_2
        return hess


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    def matmat_ATsA(s):
        return A.T @ s @ A

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = x.shape[0]
    grad_finite = np.zeros(n)
    orts = np.eye(n)
    for i in range(n):
        grad_finite[i] = (func(x + eps * orts[i]) - func(x)) / eps
    return grad_finite


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = x.shape[0]
    hess_finite = np.zeros((n, n))
    orts = np.eye(n)
    for i in range(n):
        for j in range(n):
            hess_finite[i, j] = (func(x + eps * orts[i] + eps * orts[j])
                                 - func(x + eps * orts[i])
                                 - func(x + eps * orts[j])
                                 + func(x)) / eps ** 2
    return hess_finite

# check Gradient and Hessian correctness
# points = 10 * np.random.randn(100,2)
# A = scipy.sparse.csr_matrix(np.random.randn(1000, 2))
# oracle = create_log_reg_oracle(A, np.random.choice(np.array([-1, 1]), size=1000, replace=True), 0.1)
# print(np.max([np.abs((grad_finite_diff(oracle.func, point, eps=1e-8) - oracle.grad(point)) / grad_finite_diff(oracle.func, point, eps=1e-8)) for point in points]))
# print(np.max([np.abs((hess_finite_diff(oracle.func, point, eps=1e-5) - oracle.hess(point)) / hess_finite_diff(oracle.func, point, eps=1e-5)) for point in points]))
