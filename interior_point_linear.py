# interior_point_linear.py


import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu, so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu


def randomLP(j, k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j, k))*20 - 10
    A[A[:, -1] < 0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k, :] @ x
    b[k:] = A[k:, :] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k, :].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    # Create function to get the values we are looking for question 1.
    F = lambda x, lam, mu: np.concatenate([A.T @ lam + mu - c, A @ x - b, np.diag(mu) @ x])
    n = len(c)
    m = len(b)

    def DF_solve(mu, x, lam):
        # Get the DF matrix
        a = np.hstack([np.zeros((n, n)), A.T, np.identity(n)])
        b = np.hstack([A, np.zeros((m, m + n))])
        c = np.hstack([np.diag(mu), np.zeros((n, m)), np.diag(x)])
        DF = np.vstack((a, b, c))

        # Get -F and associated variables
        neg_F = -F(x, lam, mu)
        nu = (x.T @ mu) / n
        sigma = 1 / 10
        e = np.ones(n)
        adder = np.concatenate([np.zeros(n + m), sigma * nu * e])
        right = neg_F + adder

        # Solve to get what we are looking for
        vals = la.lu_solve(la.lu_factor(DF), right)
        return vals[:n], vals[n:n + m], vals[n + m:]

    def step_size(mu, x, lam):
        # Get the results from DF_solve
        x_tri, lam_tri, mu_tri = DF_solve(mu, x, lam)

        # Get masks
        alph_mask = mu_tri < 0
        delt_mask = x_tri < 0

        # Check if mu_tri < 0:
        alph_max = np.min(-mu[alph_mask] / mu_tri[alph_mask])

        # Check if x_tri < 0:
        delt_max = np.min(-x[delt_mask] / x_tri[delt_mask])

        # Find the actual alpha, delta
        alpha = min(1, .95 * alph_max) if np.any(mu_tri) else .95
        delta = min(1, .95 * delt_max) if np.any(x_tri) else .95

        # Calculate the step sizes
        x_step = delta * x_tri
        lam_step = alpha * lam_tri
        mu_step = alpha * mu_tri

        nu = (x.T @ mu) / n

        return x_step, lam_step, mu_step, nu

        # Get initial point

    x0, lam0, mu0 = starting_point(A, b, c)

    for k in range(niter):
        x, lam, mu, nu = step_size(mu0, x0, lam0)

        # Update variables
        x0 += x
        mu0 += mu
        lam0 += lam

        # See if we need to break early
        if nu < tol:
            break

    x, lam, mu, nu = step_size(mu0, x0, lam0)
    F = lambda x, lam, mu: np.array([A.T @ lam + mu - c, A @ x - b, np.diag(mu) @ x])
    return x0, F(x, lam, mu)


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    data = np.loadtxt(filename)

    # initialize the vectors c and y
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3 * m + 2 * (n + 1))
    c[:m] = 1
    y = np.empty(2 * m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]

    # Create the constraint matrix
    A = np.ones((2 * m, 3 * m + 2 * (n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m + n] = -x
    A[1::2, m:m + n] = x
    A[::2, m + n:m + 2 * n] = x
    A[1::2, m + n:m + 2 * n] = -x
    A[::2, m + 2 * n] = -1
    A[1::2, m + 2 * n + 1] = -1
    A[:, m + 2 * n + 2:] = -np.eye(2 * m, 2 * m)

    # Calculate the solution
    sol = interiorPoint(A, y, c, niter=10)[0]

    # Extract values of beta and b
    beta = sol[m:m + n] - sol[m + n:m + 2 * n]
    b = sol[m + 2 * n] - sol[m + 2 * n + 1]

    # plot the LAD
    plt.plot(x, beta * x + b, label='LAD')

    # plot the least squares
    slope, intercept = linregress(data[:, 1], data[:, 0])[:2]
    domain = np.linspace(0, 10, 200)
    plt.plot(domain, domain * slope + intercept, label='Least Squares')

    # plot the data points
    plt.scatter(data[:, 1], data[:, 0], label='Data Points')

    plt.title("Problem 5")
    plt.tight_layout()
    plt.legend()
    plt.show()
