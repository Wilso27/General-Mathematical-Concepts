# quassian_quadrature.py


import numpy as np
from scipy import sparse as sp
import scipy.linalg as la
import scipy.stats as ss
import scipy.integrate as si
import matplotlib.pyplot as plt


class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        # check for correct polytype
        if polytype not in ['legendre','chebyshev']:
            raise ValueError('Invalid polytype')

        # store correct weight function
        if polytype == 'legendre':
            self.w_recip = lambda x: 1
        elif polytype == 'chebyshev':
            self.w_recip = lambda x: np.sqrt(1 - x**2)
        
        # store other attributes
        self.n = n
        self.polytype = polytype
        self.X, self.W = self.points_weights(self.n)


    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        # create beta array for Jacobi Matrix for given polynomial basis
        if self.polytype == 'legendre':
            beta = np.sqrt(np.array([k**2/(4*k**2 - 1) for k in range(1, n)]))
        elif self.polytype == 'chebyshev':
            beta = np.ones(n-1)/4
            beta[0] += .25
            beta = np.sqrt(beta)

        #create the matrix and get eigenvalues and eigenvectors
        J = sp.diags([beta,beta],[-1,1])
        eigvals,eigvecs = la.eigh(J.toarray())
        X = eigvals

        # use the correct mu given the poly basis
        if self.polytype == 'legendre':
            W = 2*eigvecs[0]**2
        elif self.polytype == 'chebyshev':
            W = np.pi*eigvecs[0]**2

        return X,W

    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        # use the equation in the lab
        approx = f(self.X) * self.w_recip(self.X) * self.W
        approx_sum = sum(approx)
        return approx_sum

    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        # create h and use (10.2)
        h = lambda x: f(((b-a)/2)*x + (a + b)/2) 
        return ((b - a)/2) * self.basic(h)


    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        # create a function to compute the inner sum of (10.5)
        def basic2d(f,i):
            """Approximate the integral of a f on the interval [-1,1]."""
            # use the equation in the lab
            approx = f(np.array([self.X[i]]*len(self.X)),self.X) * self.w_recip(self.X) * self.w_recip(np.array([self.X[i]]*len(self.X))) * self.W
            approx_sum = sum(approx)
            return approx_sum

        #create h for (10.5)
        h = lambda x,y: f(((b1 - a1)/2)*x + (a1 + b1)/2, ((b2 - a2)/2)*y + (a2 + b2)/2)

        approx = np.array(())
        # print(self.W[0] * basic2d(h))
        for i in range(self.n):
            approx = np.append(approx, self.W[i] * basic2d(h,i))


        return ((b1 - a1) * (b2 - a2) / 4) * sum(approx)


def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    # define the standard normal distribution
    f =lambda x: 1/(np.sqrt(2*np.pi))*np.exp(-(x**2)/2)

    # initialize arrays for storage
    legendre_approx = np.array(())
    chebyshev_approx = np.array(())

    legendre_error = np.array(())
    chebyshev_error = np.array(())

    # create the different n values
    N = np.arange(5,51,5)

    # run for each value of n
    for n in N:
        # create legendre and chebyshev objects
        GQL = GaussianQuadrature(n)
        GQC = GaussianQuadrature(n,polytype='chebyshev')

        # approximate using GaussianQuadrature class
        temp_legendre = GQL.integrate(f,-3,2)
        temp_chebyshev = GQC.integrate(f,-3,2)

        # append approximation to array
        legendre_approx = np.append(legendre_approx, temp_legendre)
        chebyshev_approx = np.append(chebyshev_approx, temp_chebyshev)

        # calculate the error
        error_legendre = abs(temp_legendre - ss.norm.cdf(2) + ss.norm.cdf(-3))
        error_chebyshev = abs(temp_chebyshev - ss.norm.cdf(2) + ss.norm.cdf(-3))

        # append error to array
        legendre_error = np.append(legendre_error, error_legendre)
        chebyshev_error = np.append(chebyshev_error, error_chebyshev)
    
    #plot the 3 things
    plt.plot(N,legendre_error,label='Legendre')
    plt.plot(N,chebyshev_error,label='Chebyshev')
    plt.axhline(si.quad(f,-3,2)[0] - ss.norm.cdf(2) + ss.norm.cdf(-3),label='Quad',color='y')
    plt.legend()
    plt.title('Errors of different integration approximation methods')
    plt.xlabel('n value')
    plt.ylabel('error value')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
