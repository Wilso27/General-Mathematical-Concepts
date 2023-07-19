# policy_iteration.py


import numpy as np
import gym

# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

# P = {s : {a: [] for a in range(4)} for s in range(4)}
# P[0][0] = [(0, 0, 0, False)]
# P[0][1] = [(1, 2, -1, False)]
# P[0][2] = [(1, 1, 0, False)]
# P[0][3] = [(0, 0, 0, False)]
# P[1][0] = [(1, 0, -1, False)]
# P[1][1] = [(1, 3, 1, True)]
# P[1][2] = [(0, 0, 0, False)]
# P[1][3] = [(0, 0, 0, False)]
# P[2][0] = [(0, 0, 0, False)]
# P[2][1] = [(0, 0, 0, False)]
# P[2][2] = [(1, 3, 1, True)]
# P[2][3] = [(1, 0, 0, False)]
# P[3][0] = [(0, 0, 0, True)]
# P[3][1] = [(0, 0, 0, True)]
# P[3][2] = [(0, 0, 0, True)]
# P[3][3] = [(0, 0, 0, True)]


P = {s : {a: [(0,0,0,False)] for a in range(4)} for s in range(6)}
P[0][3] = [(1, 1, .1, False)]
P[0][2] = [(1, 3, -1, False)]
P[1][1] = [(1, 0, -1, False)]
P[1][3] = [(1, 2,  0, False)]
P[1][2] = [(1, 4, -1, False)]
P[2][1] = [(1, 1,  -1, False)]
P[2][2] = [(1, 5,  2, True)]
P[3][0] = [(1, 0, -1, False)]
P[3][3] = [(1, 4, -1, False)]
P[4][0] = [(1, 1, .1, False)]
P[4][1] = [(1, 3, -1, False)]
P[4][3] = [(1, 5,  2, True)]
P[5][1] = [(0, 5,  0, True)]


def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    V_old = np.zeros(nS)
    V_new = np.zeros(nS)

    for iter in range(maxiter):  # each iteration of the Value function
        for s in range(nS):  # loop through each state
            sa_vector = np.zeros(nA)
            for a in range(nA):  # loop through each action at each state
                for tuple_info in P[s][a]:
                    # tuple_info is a tuple of (probability, next state, reward, done)
                    p, s_, u, _ = tuple_info
                    # sums up the possible end states and rewards with given action
                    sa_vector[a] += (p * (u + beta * V_old[s_]))
            # add the max value to the value function
            V_new[s] = np.max(sa_vector)

        # check the tolerance
        if np.linalg.norm(V_new - V_old) < tol:
            break
        else:  # update V and iterate
            V_old = V_new.copy()

    return V_new, iter + 1 # np.array([3, 3, 2, 0, 0, 0]).astype(float) #   # vector V*, num of iterations


def extract_policy(P, nS, nA, v, beta=1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    policy = np.zeros(nS, dtype=int)

    for s in range(nS):  # loop through each state
        sa_vector = np.zeros(nA)
        for a in range(nA):  # loop through each action at each state
            for tuple_info in P[s][a]:
                # tuple_info is a tuple of (probability, next state, reward, done)
                p, s_, u, _ = tuple_info
                # sums up the possible end states and rewards with given action
                sa_vector[a] += (p * (u + beta * v[s_]))
        # add the max value to the value function
        policy[s] = np.argmax(sa_vector)

    return policy  # vector V*, num of iterations


def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """

    V_old = np.zeros(nS)
    V_new = np.zeros(nS)
    maxiter = 10000

    for i in range(maxiter):

        for s in range(nS):  # loop through each state
            sa_value = 0
            for tuple_info in P[s][int(policy[s])]:
                # tuple_info is a tuple of (probability, next state, reward, done)
                p, s_, u, _ = tuple_info
                # sums up the possible end states and rewards with given action
                sa_value += (p * (u + beta * V_old[s_]))
            # add the max value to the value function
            V_new[s] = sa_value

        # check the tolerance
        norm = np.linalg.norm(V_new - V_old)
        if norm < tol:
            break
        V_old = V_new.copy()

    return V_new  # vector V*


def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """

    pik = np.random.choice(nA, size=nS)

    for k in range(maxiter):  # loop through each iteration
        vk_1 = compute_policy_v(P, nS, nA, pik, beta, tol)
        pik_1 = extract_policy(P, nS, nA, vk_1, beta)

        if np.linalg.norm(pik_1 - pik) < tol:
            break  # break after tolerance is reached
        pik = pik_1.copy()

    return vk_1, pik_1, k


def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """
    # Use basic_case boolean to determine environment
    env_name = "FrozenLake-v1" if basic_case else "FrozenLake8x8-v1"

    env = gym.make(env_name).env
    # Find number of states and actions
    nS = env.observation_space.n
    nA = env.action_space.n

    # Get the dictionary with all the states and actions
    dictionary_P = env.P

    pi_policy, pi_val, _ = policy_iteration(dictionary_P, nS, nA)
    vi_policy = extract_policy(dictionary_P, nS, nA, pi_val)[0]


    return vi_policy #(pol_iter_policy, pol_iter_val)


def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    # Use basic_case boolean to determine environment
    env_name = "FrozenLake-v1"

    env = gym.make(env_name).env
    # Find number of states and actions
    nS = env.observation_space.n
    nA = env.action_space.n

    # Get the dictionary with all the states and actions
    dictionary_P = env.P

    pi_policy, pi_val, _ = policy_iteration(dictionary_P, nS, nA)
    vi_policy = extract_policy(dictionary_P, nS, nA, pi_val)[0]

    return

