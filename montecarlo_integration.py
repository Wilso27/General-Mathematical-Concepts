# montecarlo_integration.py


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy import stats


def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    # create N n dimensional tuples
    points = np.random.uniform(-1, 1, (n, N))

    # take norms
    norms = la.norm(points, axis=0, ord=2)

    # check what is in the circle
    mask = norms < 1

    # take ratio
    ratio = np.count_nonzero(mask) / N
    return ratio * 2**n


def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    # Calculate volume
    v = b-a
    points = np.random.uniform(a, b, N)  # draw random points
    return v * np.sum(f(points)) / N  # sum f of the points


def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    # make them arrays
    mins = np.array(mins)
    maxs = np.array(maxs)

    # get the volume
    dim = len(mins)
    v = np.prod(maxs - mins)

    # get N random points in the correct dimension and shift them
    points = np.random.uniform(0, 1, (dim, N))
    shifted = points.T * (maxs - mins) + mins

    # get f values
    y = np.array([f(p) for p in shifted])

    # sum over Y values
    return v * np.sum(y) / N


def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    # create bounds and function
    mins = np.array([-1.5, 0, 0, 0])
    maxs = np.array([.75, 1, .5, 1])
    def f(x): return (1 / (2 * np.pi) ** 2) * np.exp(-(x.T @ x) / 2)

    # initialize variables
    N = np.logspace(1, 5, 20)
    rel_error = np.array(())

    # get exact value
    means, cov = np.zeros(4), np.eye(4)
    F = stats.mvn.mvnun(mins, maxs, means, cov)[0]

    for n in N:
        # loop through each N val
        F_ = mc_integrate(f, list(mins), list(maxs), int(n))
        rel_error = np.append(rel_error, np.abs(F - F_) / np.abs(F))

    # Plot the 2 curves
    plt.loglog(N, rel_error, label='Relative Error')
    plt.loglog(N, 1/np.sqrt(N), label='1/sqrt(N)')
    plt.title('Relative Error')

    plt.legend()
    plt.tight_layout()
    plt.show()
