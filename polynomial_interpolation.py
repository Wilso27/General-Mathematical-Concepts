# polynomial_interpolation.py


import numpy as np
from scipy.interpolate import BarycentricInterpolator as baryint
import scipy.linalg as la
import matplotlib.pyplot as plt


def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    #create the denominators
    L_denom = np.array([])

    #loop through each L_j
    for j in range(len(xint)):
        L_denom_j = np.array([])
        val = 1

        #loop through each k != j
        for k in range(len(xint)):

            # for k != j
            if j != k:
                val = val * (xint[j] - xint[k])
        
        #append the array
        L_denom = np.append(L_denom,val)
    
    #create n x m return matrix
    returnMatrix = np.zeros((len(xint),len(points)))

    #loop through each value in points for numerator
    for i in range(len(points)):
        x = points[i]
        for j in range(len(xint)):
            returnval = 1
            for k in range(len(xint)):
                if j != k:
                    returnval = returnval * (x - xint[k])

            #add it to the return matrix
            returnMatrix[j][i] = returnval / L_denom[j]
    
    #perform the matrix multiplication
    return yint @ returnMatrix
       

class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        # Attributes
        self.x = xint
        self.y = yint
        
        n = len(xint)
        # Given a NumPy array xint of interpolating x-values, calculate the weights.
        w = np.ones(n) # Array for storing barycentric weights.
        # Calculate the capacity of the interval.
        C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(n-1)
        for j in range(n):
            temp = (xint[j] - np.delete(xint, j)) / C
            temp = temp[shuffle] # Randomize order of product.
            w[j] /= np.product(temp)
        self.weights = w

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        eval = np.array([])
        for x in points:
            if x in self.x:
                eval = np.append(eval, self.y[np.where(self.x == x)])
                continue
            poly_num = np.sum((self.weights * self.y) / (x - self.x))
            poly_denom = np.sum(self.weights / (x - self.x))
            eval = np.append(eval, poly_num/poly_denom)
        return eval
    
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        self.x = np.append(self.x,xint)
        self.y = np.append(self.y,yint)
        n = len(self.x)

        self.weights = np.ones(n) # Array for storing barycentric weights.
        # Calculate the capacity of the interval.
        C = (np.max(self.x) - np.min(self.x)) / 4
        shuffle = np.random.permutation(n-1)
        for j in range(n):
            temp = (self.x[j] - np.delete(self.x, j)) / C
            temp = temp[shuffle] # Randomize order of product.
            self.weights[j] /= np.product(temp)


def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    #initialize variables
    x = np.linspace(-1,1,400)
    abs_err = np.array([])
    cheby_abs_err = np.array([])
    f = lambda x: 1/(1 + 25*x**2)
    for i in range(2,9):
        #loop through each n value
        n = 2**i
        xint = np.linspace(-1,1,n)
        yint = f(xint)
        baryapprox = baryint(xint,yint)

        # compute the infty norm of the original
        abs_err = np.append(abs_err, la.norm(f(x) - baryapprox(x), ord=np.inf))
        
        # compute the err of the chebyshev 
        cheby = .5 * (np.min(xint) + np.max(xint) + ((np.max(xint) - np.min(xint)) * np.cos(np.arange(0,n+1,1)*np.pi/n)))
        cheby_bary = baryint(cheby,f(cheby))
        cheby_abs_err = np.append(cheby_abs_err, la.norm(f(x) - cheby_bary(x), ord=np.inf))
    
    # plot on loglog plot
    plt.loglog(np.array([2**i for i in range(2,9)]), abs_err - cheby_abs_err)
    plt.loglog(np.array([2**i for i in range(2,9)]), cheby_abs_err)

    plt.show()

        
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    # get cheby points
    y = np.array([np.cos(j*np.pi/n) for j in range(2*n)])

    # get DFT values
    DFT = np.real(np.fft.fft(f(y))/(2*n))

    # create gamma coefficients
    gamma = np.array([1] + [2 for i in range(n - 1)] + [1])

    # compute the a_k's 
    a = gamma * DFT[:n+1]

    return a


def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    #import data
    data = np.load('airdata.npy')

    #use code given in lab to get the points nearest the cheby points
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    poly = baryint(domain[temp2], data[temp2])

    #plot original data and int poly on subplots
    plt.subplot(211)
    plt.title('Original Data')
    plt.plot(data)

    plt.subplot(212)
    plt.title(f'Interpolating Polynomial of Degree n={n}')
    plt.plot(poly(domain))

    plt.tight_layout()
    plt.show()
    

def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        self.x = np.append(self.x,xint)
        self.y = np.append(self.y,yint)
        n = len(self.x)

        w = np.ones(n) # Array for storing barycentric weights.
        # Calculate the capacity of the interval.
        C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(n-1)
        for j in range(n):
            temp = (self.x[j] - np.delete(self.x, j)) / C
            temp = temp[shuffle] # Randomize order of product.
            w[j] /= np.product(temp)
        self.weights = w


        # Store the x and y values of the interpolating points and the number of points and C.
        self.x = np.append(self.x, xint)
        self.y = np.append(self.y, yint)
        self.n += len(xint)
        self.C = (np.max(self.x) - np.min(self.x)) / 4
        # Initialize the weights attribute, shuffle the x values, and loop through the x values.
        self.w = np.ones(self.n)
        shuffle = np.random.permutation(self.n - 1)
        for i in range(self.n):
            # Calculate scaled weights and store them in the weights attribute.
            temp = (self.x[i] - self.x[np.arange(self.n) != i]) / self.C
            temp = temp[shuffle]
            self.w[i] /= np.prod(temp)

