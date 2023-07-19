

import numpy as np


class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        # store dimensions as attributes
        self.m, self.n = A.shape[0], A.shape[1]

        # create the origin vector
        x = np.zeros(self.n)
        mask = A@x <= b  # mask it to get array of 0's and 1's
        if np.sum(mask) != self.m:  # sum array to see if all not true
            raise ValueError("Origin is not feasible")

        # store dictionary using method
        self.dictionary = self._generatedictionary(c, A, b)

    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        # create the elements of D by row
        A_bar = np.hstack((A, np.eye(self.m)))
        row1 = np.hstack((np.zeros(1), c, np.zeros(self.m)))
        b_ = np.vstack(b)
        row2 = np.hstack((b_, -A_bar))

        # combine elements into D
        D = np.vstack((row1, row2))

        return D

    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        # create mask and return the first true
        mask = self.dictionary[0, 1:] < 0
        return np.argmax(mask) + 1

    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        # create mask and return the abs val min of the ratios
        pivot_col = self.dictionary[1:, index]
        mask = pivot_col < 0

        if np.sum(mask) == 0:  # check if mask is empty
            raise ValueError("The problem is unbounded")
        const_col = self.dictionary[1:, 0]

        ratio = np.array(())

        # loop through each negative val in the column to get ratio
        for y in range(len(pivot_col)):
            if pivot_col[y] < 0:
                ratio = np.append(ratio, const_col[y] / -pivot_col[y])
            else:
                ratio = np.append(ratio, np.inf)

        # return the argmin
        return np.argmin(ratio) + 1

    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """

        # Get the pivot row and column
        j = self._pivot_col()
        i = self._pivot_row(j)

        # create a -1 in the i,j entry
        k = self.dictionary[i][j]
        self.dictionary[i] /= -k

        # Loop through each row other than i
        for x in range(self.m+1):
            if x != i:
                self.dictionary[x] += self.dictionary[i] * self.dictionary[x][j]

    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        # pivot until solved
        while np.any(self.dictionary[0,1:] < 0):
            self.pivot()

        dep_vars = {}
        ind_vars = {}

        # get the values from first column
        const_col = self.dictionary[:, 0]

        # CHECK OBJ ROW TO SEE IF THEY'RE NONZERO AND THOSE ARE DEP, ELSE IND
        for n in range(1, len(self.dictionary[0])):
            if self.dictionary[0][n] == 0:
                val = np.argmin(self.dictionary[:, n])
                dep_vars[n-1] = const_col[val]
            else:  # if ind set to zero
                ind_vars[n-1] = 0

        return self.dictionary[0][0], dep_vars, ind_vars


def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    # load in data
    data = np.load(filename)
    A = data['A']
    p = data['p']
    m = data['m']
    d = data['d']

    n = len(p)

    c = -p  # make it minimization
    b = np.concatenate((m, d))

    B = np.vstack((A, np.eye(n)))

    # solve using function
    simp = SimplexSolver(c, B, b)
    sol = simp.solve()

    products = np.zeros(n)

    # pull values for independent and dependent variables
    for i in range(n):
        if i in sol[1]:
            products[i] = sol[1][i]
        else:
            products[i] = sol[2][i]

    return products
