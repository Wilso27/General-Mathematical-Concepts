# dynamic_programming.py


import numpy as np
import matplotlib.pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    # initialize array to store
    V = np.array([0])
    for t in np.arange(N, 0, -1):  # loop through each t value
        # use the algorithm given
        V = np.append(V, max((t-1)/t * V[-1] + (1/N), V[-1]))
    t0 = np.argmax(V)  # find the optimal stopping point
    return V[t0], N - t0  # return the optimal value and stopping point


def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    # initialize variables
    osp_list = []
    V_list = []
    domain = np.arange(3, M+1)
    for N in domain:  # loop through each N value
        V, t0 = calc_stopping(N)
        V_list.append(V)
        osp = t0 / N
        osp_list.append(osp)

    # plot the function
    plt.title("Problem 2")
    plt.plot(domain, V_list, label="Optimal Stopping Percentage")
    plt.plot(domain, osp_list, label="Maximum Probability")
    plt.legend()
    plt.show()

    return osp_list[-1]


def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    # initialize variables
    bites = np.arange(0, N+1)[::-1] / N
    C = np.zeros((N+1, N+1))

    for i in range(N+1):  # loop through each row
        pad_bites = np.pad(bites[N - i:], (0, N - i), 'constant',
                         constant_values=(0, 0))
        C[i,:] = pad_bites

    return u(C)  # return consumption matrix


def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    # initialize variables
    A = np.zeros((N + 1, T + 1))
    P = np.zeros((N + 1, T + 1))
    w = np.linspace(0, 1, N+1)

    # add the last columns
    A[:, T] = u(w)
    P[:, T] = w

    for t in range(T, 0, -1):  # for each t
        CV = np.zeros((N+1, N+1))
        for j in range(N+1):  # loop through columns and rows
            for i in range(N+1):
                if j <= i:
                    CV[i, j] = u(w[i] - w[j]) + B * A[j, t]
                A[i, t - 1] = np.max(CV[i])

                # update P matrix
                P[i, t-1] = w[i] - w[np.argmax(CV[i])]
    return A, P


def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    # use previous function
    P = eat_cake(T, N, B, u)[1]
    w = np.linspace(0, 1, N + 1)
    c = []

    # initialize variables
    row = N
    j = 0
    slices = 1

    while slices >= 0 and j < T+1:  # loop through T times
        if slices == 0:  # bite is 0 if cake is 0
            c.append(0)
        else:  # fix if P[row, j] returns a float instead of array
            c.append(P[row, j][0]) if type(P[row, j]) is np.ndarray else c.append(P[row, j])
        row -= np.where(w == c[j])[0]
        slices -= c[j]
        j += 1

    return c  # return the policy
