
"""
Implementation of GMM
__author__ = "Mehrdad Tamiji"
__email__ = "mehrdad.tamiji@gmail.com"
"""

import numpy as np
def alpha_pass(alpha, c, observations, observ_seq, init_state_prob, emission_matrix, transition_matrix, Len_observ_seq):
    """
    alpha pass or forward
    """
    current_observ = observations[observ_seq[0]]
    alpha[0] = init_state_prob * emission_matrix[:, current_observ]
    c[0] = np.sum(alpha[0])
    alpha[0] = alpha[0] / np.sum(alpha[0])
    for t in range(1, Len_observ_seq):
        alpha[t] = alpha[t - 1] @ transition_matrix
        current_observ = observations[observ_seq[t]]
        alpha[t] *= emission_matrix[:, current_observ]
        assert np.sum(alpha[t]) != 0
        c[t] = np.sum(alpha[t])
        alpha[t] = alpha[t] / np.sum(alpha[t])
    return alpha, c

def beta_pass(beta, c, observations, observ_seq, emission_matrix, transition_matrix, Len_observ_seq, num_states):
    """
    beta pass or backward
    """
    beta[Len_observ_seq - 1] = np.ones((num_states))
    for t in range(Len_observ_seq-2, -1, -1):
        next_observ = observations[observ_seq[t+1]]
        beta[t] = transition_matrix @ (emission_matrix[:, next_observ] * beta[t + 1].T) #(transition_matrix[0,0] * emission_matrix[0, next_observ]* beta[t,0]) + (transition_matrix[0,1] * emission_matrix[1, next_observ]* beta[t,1])
        beta[t] = beta[t] / c[t]
    return beta

def calc_gammas(alpha, beta, digm, gamma, observations, observ_seq, emission_matrix, transition_matrix, Len_observ_seq, num_states):
    """
    Calculate gamma and digamma which are used for training the model.
    """
    for t in range(Len_observ_seq - 1):
        for i in range(num_states):
            gamma[t, i] = 0
            next_observ = observations[observ_seq[t + 1]]
            for j in range(num_states):
                digm[t, i, j] = alpha[t, i] * transition_matrix[i, j] * emission_matrix[j, next_observ] * beta[t + 1, j] # (alpha[0,0]*transition_matrix[0,0]*emission_matrix[0, next_observ]*beta[t,0])+ (alpha[0,0]*transition_matrix[0,1]*emission_matrix[1, next_observ]*beta[t,1])
                gamma[t, i] += digm[t, i, j]
    gamma[-1] = alpha[-1]
    return digm, gamma

def train(digm, gamma, num_observables, observ_seq_int, emission_matrix, transition_matrix, Len_observ_seq, num_states):
    """
    Train our model by estimating transition and emission matrix
    """
    init_state_prob = gamma[0]
    # calc transition
    for i in range(num_states):
        denom = np.sum(gamma[:-1, i])
        for j in range(num_states):
            numer = np.sum(digm[:-1, i, j])
            transition_matrix[i, j] = numer / denom
        transition_matrix[i] /= np.sum(transition_matrix[i])
    # calc emission
    for i in range(num_states):
        denom = np.sum(gamma[:, i])
        for j in range(num_observables):
            numer = np.sum(gamma[observ_seq_int==j, i])
            emission_matrix[i, j] = numer / denom
        emission_matrix[i] /= np.sum(emission_matrix[i])
    logProb = 0
    for i in range(Len_observ_seq):
        logProb+= np.log(c[i])
    logProb = -logProb
    return init_state_prob, transition_matrix, emission_matrix, logProb


def generate_sequence(lenght, observ_seq):
    """
    Generating a sequence based on ground truth
    """
    current_state = states[np.random.choice(list(states.keys()), 1, p=ground_truth_init_state_prob)[0]]
    for i in range(lenght):
        observ_seq += np.random.choice(list(observations.keys()), 1, p=ground_truth_emission_matrix[current_state])[0]
        previous_state = current_state
        current_state = states[np.random.choice(list(states.keys()), 1, p=ground_truth_transition_matrix[previous_state])[0]]
    return observ_seq

def checking(ground_truth_transition_matrix, ground_truth_emission_matrix, transition_matrix, emission_matrix):
    print("-------------------final checking---------------------------")
    print("transition:\n", transition_matrix)
    print("emission:\n", emission_matrix)
    diff_transition = ground_truth_transition_matrix[ground_truth_transition_matrix[:, 0].argsort()[::-1]] - transition_matrix[transition_matrix[:, 0].argsort()[::-1]]
    diff_emission = ground_truth_emission_matrix[ground_truth_emission_matrix[:, 0].argsort()[::-1]] - emission_matrix[emission_matrix[:, 0].argsort()[::-1]]
    print("difference transition:\n", diff_transition)
    print("difference emission:\n", diff_emission)


if __name__ == '__main__':
    num_states =  2
    num_observables = 4
    states = {"A": 0, "B":1 }
    observations =  {"A": 0, "C":1, "G":2, "T":3 }
    ground_truth_transition_matrix = np.array([[.7, .3], [.3, .7]])
    ground_truth_emission_matrix = np.array([[.3, .2, .2, .3], [.05, .05, .45, .45]]) # in exercise file, second row has a problem.
    ground_truth_init_state_prob = np.array([.5, .5])
    observ_seq = ""
    observ_seq = generate_sequence(2000, observ_seq)
    observ_seq_int = np.array([observations[i] for i in observ_seq])
    init_state_prob = np.array([.6, .4])
    transition_matrix = np.array([[.4, .6], [.6, .4]])
    emission_matrix = np.array([[.1, .25, .50, .15], [.5, .25, .15, .1]])
    Len_observ_seq = len(observ_seq)
    prob_observ_seq = 1
    oldLogProb = -np.inf
    epochs = 500
    for epoch in range(epochs):
        if epoch % 10 == 0: print("epoch", epoch)
        alpha = np.zeros((Len_observ_seq, num_states))
        beta = np.zeros((Len_observ_seq, num_states))
        c = np.zeros(Len_observ_seq)
        alpha, c = alpha_pass(alpha, c, observations, observ_seq, init_state_prob, emission_matrix, transition_matrix, Len_observ_seq)
        beta = beta_pass(beta, c, observations, observ_seq, emission_matrix, transition_matrix, Len_observ_seq, num_states)
        digm = np.zeros((Len_observ_seq, num_states, num_states))
        gamma = np.zeros((Len_observ_seq, num_states))
        digm, gamma = calc_gammas(alpha, beta, digm, gamma, observations, observ_seq, emission_matrix, transition_matrix, Len_observ_seq, num_states)
        est_transition_matrix = np.zeros((num_states, num_states))
        est_emission_matrix = np.zeros((num_states, num_observables))
        init_state_prob, transition_matrix, emission_matrix, logProb = train(digm, gamma, num_observables, observ_seq_int, emission_matrix, transition_matrix, Len_observ_seq, num_states)
        if logProb > oldLogProb:
            oldLogProb = logProb
            np.set_printoptions(precision=5)
            print("transition:\n", transition_matrix)
            print("emission:\n", emission_matrix)
            print("satisfied")
        if epoch % 50 ==0:
            np.set_printoptions(precision=5)
            print("transition:\n", transition_matrix)
            print("emission:\n", emission_matrix)
    checking(ground_truth_transition_matrix, ground_truth_emission_matrix, transition_matrix, emission_matrix)