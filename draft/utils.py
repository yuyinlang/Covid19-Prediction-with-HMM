# Testing module
import numpy as np
def test_hmm(test_module, hmm):

    # Give testing sequence here    
    obs_seq1 = np.array([[0.9, 0.2], [7.0, 1.2], [11.0, 0.01], [0.33, 0.45], [0.72, 0.33]])
    obs_seq2 = np.array([[0.1, 0.02], [0.12, 0.3], [9.1, 0.1], [0.93, 0.15], [1.72, 0.03]])
    obs_seq3 = np.array([[0.1, 0.1], [0.02, 0.8], [1.8, 1.8], [0.0, 0.15], [1.0, 0.03]])
    obs_seqs = [obs_seq1, obs_seq2, obs_seq3]
    
    saved_results = np.load('test_examples.npz',fix_imports=True, encoding='latin1', allow_pickle=True)['data']

    N_correct = 0
    
    for i, obs_seq in enumerate(obs_seqs):
        result = saved_results[i]
        smoothed_prob = result['smoothing'] 
        predicted_prob = result['prediction']
        trajectory = result['trajectory'] 
        
        if test_module == 'forward_backward':
            smoothed_prob_test = hmm.forward_backward(obs_seq)
            assert smoothed_prob_test.shape == smoothed_prob.shape
            
            if np.allclose(smoothed_prob_test, smoothed_prob): 
                N_correct += 1
                    
        elif test_module == 'prediction':
            predicted_prob_test = hmm.predict(obs_seq, 10)
            assert predicted_prob_test.shape == predicted_prob.shape
            if np.allclose(predicted_prob_test, predicted_prob): 
                N_correct += 1
                    
        elif test_module == 'viterbi':
            trajectory_test = hmm.viterbi(obs_seq)
            assert trajectory_test.shape == trajectory.shape
            if np.allclose(trajectory_test, trajectory): 
                N_correct += 1
                    
        else:
            raise NotImplementedError
            
    if N_correct == len(obs_seqs):
        print('Congratulations! Your implementation passed all testing examples for {} test!'.format(test_module))
    else:
        print('Sorry, Your implementation for {} is not correct, only passed {}/{} of the testing examples :('.format(test_module, N_correct, len(obs_seqs)))

    
# # Test

# # Let's define we have N states, and our observation is a vector with H dimension.
# # In our scenario, N = 7, H = 2.

# # Input transition matrix, NxN matrix in our scenario (N states)
# T = np.array([[0.2, 0.6],
#               [0.8, 0.4]])

# # Input mean of gaussian variable here, Nx2 matrix in our scenario,
# # Since each state has its own corresponding gaussian distribution (N states), and the dimension of the observation is H.
# # For each H-dimensional gaussian distribution, its corresponding mean (mu) vector shall has H dimension.
# # So we input 7xH matrix as *mus* in our scenario.
# mus = np.array([[0.8, 0.7], 
#                 [0.3, 0.2]])

# # Input mean of gaussian variable here, NxHxH matrix in our scenario,
# # Since each state has its own corresponding gaussian distribution (N states), and the dimension of the observation is H.
# # For each H-dimensional gaussian distribution, its corresponding covariance matrix (sigma) shall be HxH.
# # So we input NxHxH matrix as *sigmas* in our scenario.
# sigmas = np.zeros((2, 2, 2))
# sigmas[0] = np.eye(2)
# sigmas[1] = np.eye(2)

# # Input initialized state distribution here, a vector with N dimension. (N states)
# init_prob = np.array([0.3, 0.7])

# # Input observation sequence here, a TxH matrix. In this scenario, we have data with T timestamps.
# # And each row is a observation vector with H dimension. 
# obs_seq1 = np.array([[0.9, 0.2], [7.0, 1.2], [11.0, 0.01], [0.33, 0.45], [0.72, 0.33]])
# obs_seq2 = np.array([[0.1, 0.02], [0.12, 0.3], [9.1, 0.1], [0.93, 0.15], [1.72, 0.03]])
# obs_seq3 = np.array([[0.1, 0.1], [0.02, 0.8], [1.8, 1.8], [0.0, 0.15], [1.0, 0.03]])
# obs_seqs = [obs_seq1, obs_seq2, obs_seq3]
# saved_results = []

# for i, obs_seq in enumerate(obs_seqs):
#     # Test Forward-backward algorithm
#     hmm = HMM(mus, sigmas, T, init_prob)
#     smoothed_prob = hmm.forward_backward(obs_seq)

#     # Test Prediction
#     predicted_prob = hmm.predict(obs_seq, 10)

#     # Test viterbi algorithm
#     trajectory = hmm.viterbi(obs_seq)
    
#     result = {}
#     result['smoothing'] = smoothed_prob
#     result['prediction'] = predicted_prob
#     result['trajectory'] = trajectory
    
#     saved_results.append(result)

# for i in range(len(saved_results)):
#     result = saved_results[i]
#     smoothed_prob = result['smoothing'] 
#     predicted_prob = result['prediction']
#     trajectory = result['trajectory'] 
    
#     print('smoothing: \n', smoothed_prob, '\n')
#     print('prediction: \n', predicted_prob, '\n')
#     print('trajectory: \n',trajectory, '\n')

# np.savez('test_examples.npz', data=saved_results)