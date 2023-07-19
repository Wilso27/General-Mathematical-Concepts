# cvxpy_intro.py


import cvxpy as cp
import numpy as np


def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # initialize the objective
    x = cp.Variable(3, nonneg=True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    # write the constraints
    A = np.array([1, 2, 0])
    G = np.array([0, 1, -4])
    Q = np.array([2, 10, 3])
    P = np.eye(3)
    constraints = [A @ x <= 3, G @ x <= 1, Q @ x >= 12, P @ x >= 0]  # This must be a list

    # Assemble the problem and then solve it
    problem = cp.Problem(objective, constraints)
    val = problem.solve()

    return x.value, val


def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # get dimensions and initialize variables
    m, n = A.shape
    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm(x, 1))

    constraints = []
    for i in range(m):
        constraints.append(A[i] @ x == b[i])

    # assemble the problem and solve it
    problem = cp.Problem(objective, constraints)
    val = problem.solve()

    return x.value, val


def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # get dimensions and initialize variables
    x = cp.Variable(6, nonneg=True)
    c = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(c @ x)

    A = np.array([[1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1]
    ])

    b = np.array([7, 2, 4, 5, 8])

    constraints = []
    for i in range(5):
        if i < 3:
            constraints.append(A[i] @ x <= b[i])
        else:
            constraints.append(A[i] @ x >= b[i])

    # assemble the problem and solve it
    problem = cp.Problem(objective, constraints)
    val = problem.solve()

    return x.value, val


def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # create the quadratic equation
    Q = np.array([[3, 2, 1], [2, 4, 2], [1, 2, 3]])
    r = np.array([3, 0, 1])
    x = cp.Variable(3)

    # assemble the problem
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T @ x))
    val = prob.solve()

    return x.value, val


def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # get dimensions and initialize variables
    m, n = A.shape
    x = cp.Variable(n, nonneg=True)

    objective = cp.Minimize(cp.norm(A @ x - b, 2))
    constraints = [sum(x) == 1]

    # assemble the problem and solve it
    problem = cp.Problem(objective, constraints)
    val = problem.solve()

    return x.value, val


def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # load in the file
    filename = "food.npy"
    data = np.load(filename, allow_pickle=True)

    # initialize variables
    x = cp.Variable(18, nonneg=True)
    b = np.array([2000, 65, 50, 1000, 25, 46])
    s = data[:, 1]
    p = data[:, 0]
    A = np.array([data[:, 2]])
    for j in range(3, 8):
        A = np.vstack((A, data[:, j]))

    # create the objective and constraints
    objective = cp.Minimize(p.T @ x)
    constraints = []
    for i in range(6):
        if i < 3:
            constraints.append((A[i] * s).T @ x <= b[i])
        else:
            constraints.append((A[i] * s).T @ x >= b[i])

    # assemble the problem and solve it
    problem = cp.Problem(objective, constraints)
    val = problem.solve()

    return x.value, val
