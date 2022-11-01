# Testing module
import numpy as np
from scipy.stats import multivariate_normal


def dict_to_array(state_dict):
    state_array = [state_dict[t] for t in sorted(state_dict.keys())]
    return np.array(state_array)


def get_obs_mat(observation_mu, observation_sigma):
    obs_mat = [multivariate_normal(observation_mu[i], observation_sigma[i]) for i in range(len(observation_mu))]
    return obs_mat

