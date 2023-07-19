# differentiation.py


import sympy as sy
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la
from autograd import numpy as anp
from autograd import elementwise_grad
from autograd import grad
import time


def prob1(plot=False):
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    # set up sympy variables and functions
    x = sy.symbols('x')
    f = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))

    # take the derivative
    df = sy.diff(f)

    # lambdify the functions
    df = sy.lambdify(x, df, 'numpy')
    f = sy.lambdify(x, f, 'numpy')

    if plot:
        # plot the functions on the same plot
        domain = np.linspace(-np.pi,np.pi,100)
        ax = plt.gca()
        ax.spines['bottom'].set_position('zero')
        ax.plot(domain, f(domain), label='f')
        ax.plot(domain, df(domain), label='df')
        ax.legend()
        plt.title('f versus df')
        plt.tight_layout()
        plt.show()

    # return the lambdified function
    return df


def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x + h) - f(x) ) / h


def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3*f(x) + 4*f(x + h) - f(x + 2*h)) / (2*h)


def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x) - f(x - h)) / h


def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3*f(x) - 4*f(x - h) + f(x - 2*h)) / (2*h)


def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x + h) - f(x - h)) / (2*h)


def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h)) / (12*h)


def plot_diff_quotients():
    '''Plotting like it asks us to in the previous problem'''

    # creating the sympy function
    x = sy.symbols('x')
    f = (sy.sin(x) + 1) ** (sy.sin(sy.cos(x)))
    f = sy.lambdify(x, f, 'numpy')

    # plot the functions on the same plot
    domain = np.linspace(-np.pi, np.pi, 100)

    # plot the 6 functions from prob2()
    plt.subplot(321)
    plt.plot(domain, f(domain), label='f')
    plt.plot(domain, fdq1(f,domain), label='df')
    plt.title('fdq1')
    plt.legend()

    plt.subplot(322)
    plt.plot(domain, f(domain), label='f')
    plt.plot(domain, fdq2(f, domain), label='df')
    plt.title('fdq2')
    plt.legend()

    plt.subplot(323)
    plt.plot(domain, f(domain), label='f')
    plt.plot(domain, bdq1(f, domain), label='df')
    plt.title('bdq1')
    plt.legend()

    plt.subplot(324)
    plt.plot(domain, f(domain), label='f')
    plt.plot(domain, bdq2(f, domain), label='df')
    plt.title('bdq2')
    plt.legend()

    plt.subplot(325)
    plt.plot(domain, f(domain), label='f')
    plt.plot(domain, cdq2(f, domain), label='df')
    plt.title('cdq2')
    plt.legend()

    plt.subplot(326)
    plt.plot(domain, f(domain), label='f')
    plt.plot(domain, cdq4(f, domain), label='df')
    plt.title('cdq4')
    plt.legend()

    plt.tight_layout()
    plt.show()


def prob3(x0):  # FIXED SINCE HARD GRADE
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    # new prob1() because my prob1 would plot it again when called
    def prob1_again(): # FIXED SINCE HARD GRADE
        """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""

        # set up sympy variables and functions
        x = sy.symbols('x')
        f = (sy.sin(x) + 1) ** (sy.sin(sy.cos(x)))

        # take the derivative
        df = sy.diff(f)

        # lambdify the functions
        df = sy.lambdify(x, df, 'numpy')
        f = sy.lambdify(x, f, 'numpy')

        return f, df

    # call the function to get diff
    f, df = prob1_again()

    # initialize error arrays
    fdq1_err = np.array(())
    fdq2_err = np.array(())
    bdq1_err = np.array(())
    bdq2_err = np.array(())
    cdq2_err = np.array(())
    cdq4_err = np.array(())

    # create h array
    h = np.logspace(-8, 0, 9)

    # Loop through all values of h for each method and append to individual array
    for k in range(9):
        fdq1_err = np.append(fdq1_err, abs(df(x0) - fdq1(f, x0, h[k])))

        fdq2_err = np.append(fdq2_err, abs(df(x0) - fdq2(f, x0, h[k])))

        bdq1_err = np.append(bdq1_err, abs(df(x0) - bdq1(f, x0, h[k])))

        bdq2_err = np.append(bdq2_err, abs(df(x0) - bdq2(f, x0, h[k])))

        cdq2_err = np.append(cdq2_err, abs(df(x0) - cdq2(f, x0, h[k])))

        cdq4_err = np.append(cdq4_err, abs(df(x0) - cdq4(f, x0, h[k])))

    # plot on loglog scale on same graph to make it look like lab specs
    plt.loglog(h, fdq1_err, label='Order 1 Forward')
    plt.loglog(h, fdq2_err, label='Order 2 Forward')
    plt.loglog(h, bdq1_err, label='Order 1 Backward')
    plt.loglog(h, bdq2_err, label='Order 2 Backward')
    plt.loglog(h, cdq2_err, label='Order 2 Centered')
    plt.loglog(h, cdq4_err, label='Order 4 Centered')

    plt.xlabel('Absolute Error')
    plt.ylabel('h')
    plt.legend()
    plt.title('Approximating Derivative Values')
    plt.tight_layout()
    plt.show()


def prob4():  # INCORRECT AND UNEDITED SINCE HARD GRADE FEEDBACK
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    # get the data from the file
    data = np.load('plane.npy')
    alpha = np.array(np.deg2rad(data[:, 1]))
    beta = np.array(np.deg2rad(data[:, 2]))

    # get the x and y coords
    x = 500 * (np.tan(beta)) / (np.tan(beta) - np.tan(alpha))
    y = 500 * (np.tan(beta) * np.tan(alpha)) / (np.tan(beta) - np.tan(alpha))

    # approximating x_prime
    x_prime = np.array([(x[1] - x[0])])
    x_prime = np.append(x_prime, np.zeros(len(x) - 2))
    x_prime[1:len(x) - 1] = (x[:-2] - x[2:]) / 2
    x_prime = np.append(x_prime, (x[len(x) - 1] - x[len(x) - 2]))

    # approximating y_prime
    y_prime = np.array([(y[1] - y[0])])
    y_prime = np.append(y_prime, np.zeros(len(y) - 2))
    y_prime[1:len(y) - 1] = (y[:-2] - y[2:]) / 2
    y_prime = np.append(y_prime, (y[len(y) - 1] - y[len(y) - 2]))

    # calculate the approximate speed
    speed = np.sqrt(x_prime**2 + y_prime**2)

    return speed


def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    # get dimension of Jacobian
    n = len(x)
    m = len(f(x))
    J = np.zeros((m, n))

    # get the standard basis vectors in an array of arrays
    e = np.eye(n)

    # do (8.5) for each e_j and store as matrix
    for j in range(n):
        J[:, j] = (f(x + h * e[j]) - f(x - h * e[j])) / (2 * h)

    return J


def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (jax.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    T_n = anp.array([1, x])
    for i in range(n + 1):
        T_n = anp.append(T_n, 2 * x * T_n[-1] - T_n[-2])

    if n == 0:
        return anp.ones_like(x)
    elif n == 1:
        return x
    else:
        return 2 * x * cheb_poly(x, n - 1) - cheb_poly(x, n - 2)


def prob6():
    """Use JAX and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    domain = anp.linspace(-1, 1, 100)
    df = elementwise_grad(cheb_poly)

    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.title(f"n = {i}")
        plt.plot(domain, df(domain, i))

    plt.tight_layout()
    plt.show()


def prob7(N=200):
    """
    Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the "exact" value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            JAX (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and JAX.
    For SymPy, assume an absolute error of 1e-18.
    """
    # initialize storage
    cdq4_err = np.array(())
    ag_err = np.array(())
    exact_time = np.array(())
    centered_time = np.array(())
    ag_time = np.array(())

    # set up sympy variables and functions
    x = sy.symbols('x')
    g = lambda x: (anp.sin(x) + 1) ** (anp.sin(anp.cos(x)))

    # repeat experiment N times
    for i in range(N):
        x0 = anp.random.random() * 40 - 20

        # time exact
        start = time.perf_counter()
        f = (sy.sin(x) + 1) ** (sy.sin(sy.cos(x)))
        df = sy.diff(f)
        df = sy.lambdify(x, df, 'numpy')
        f = sy.lambdify(x, f, 'numpy')
        exact = df(x0)
        end = time.perf_counter()
        exact_time = np.append(exact_time, end - start)

        # time first approx
        start = time.perf_counter()
        centered = cdq4(f, x0)
        end = time.perf_counter()
        centered_time = np.append(centered_time, end - start)

        # time second approx
        start = time.perf_counter()
        ag = grad(g)(x0)
        end = time.perf_counter()
        ag_time = np.append(ag_time, end - start)

        # record errors
        cdq4_err = np.append(cdq4_err, abs(df(x0) - cdq4(f, x0)))
        ag_err = np.append(ag_err, abs(df(x0) - ag))

    # plot the 3 of them
    plt.loglog(exact_time, np.ones(N) * 10 ** (-18), '.', label='Sympy', alpha=.3)
    plt.loglog(centered_time, cdq4_err, '.', label='Difference Quotient', alpha=.3)
    plt.loglog(ag_time, ag_err, '.', label='Autograd', alpha=.3)

    plt.title('Time vs. Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
