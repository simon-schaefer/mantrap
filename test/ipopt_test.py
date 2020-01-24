import ipopt
import numpy as np


class HS071(object):
    def __init__(self):
        pass

    @staticmethod
    def objective(x):
        return x[0] * x[3] * np.sum(x[0:3]) + x[2]

    @staticmethod
    def gradient(x):
        return np.array([
            x[0] * x[3] + x[3] * np.sum(x[0:3]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * np.sum(x[0:3])
        ])

    @staticmethod
    def constraints(x):
        return np.array((np.prod(x), np.dot(x, x)))

    @staticmethod
    def jacobian(x):
        return np.concatenate((np.prod(x) / x, 2*x))

    # def hessianstructure(self):
    #     global hs
    #     hs = scipy.sparse.coo_matrix(np.tril(np.ones((4, 4))))
    #     return hs.col, hs.row  # default is triangular matrix

    @staticmethod
    def hessian(x, lagrange, obj_factor):
        hessian = obj_factor*np.array((
            (2*x[3], 0, 0, 0),
            (x[3],   0, 0, 0),
            (x[3],   0, 0, 0),
            (2*x[0]+x[1]+x[2], x[0], x[0], 0))
        )
        hessian += lagrange[0]*np.array((
                (0, 0, 0, 0),
                (x[2]*x[3], 0, 0, 0),
                (x[1]*x[3], x[0]*x[3], 0, 0),
                (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        hessian += lagrange[1]*2*np.eye(4)
        return hessian

    @staticmethod
    def intermediate(
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials
    ):
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


if __name__ == '__main__':
    x0 = [1.0, 5.0, 5.0, 1.0]  # initial condition

    lb = [1.0, 1.0, 1.0, 1.0]  # lower bounds
    ub = [5.0, 5.0, 5.0, 5.0]  # upper bounds

    cl = [25.0, 40.0]  # constraints lower bound
    cu = [2.0e19, 40.0]  # constraints upper bound (second constraint is equality constraint !)

    nlp = ipopt.problem(
        n=len(x0),
        m=len(cl),
        problem_obj=HS071(),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )
    nlp.addOption("mu_strategy", "adaptive")
    nlp.addOption("tol", 1e-7)
    solution, info = nlp.solve(x0)

    assert np.array_equal(np.round(solution, 3), np.array([1, 4.743, 3.821, 1.379]))
