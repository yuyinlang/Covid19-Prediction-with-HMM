#!/usr/bin/env python
# coding: utf-8

# The goal of this task is to implement a Hidden Markov Model and inferring the most likely sequence of states using the Viterbi algorithm.
# Additionally, the inference tasks prediction and smoothing have to be implemented.

# We all know that COVID-19 has been spreading for more than two years by now. In order to fight the pandemic,
# many restrictions have been implemented and are changing constantly. As a result, conducting our exam on-site is
# in danger. We want to find out if we again will have to resort to an online exam.
# 
# To do so, we want to use an HMM modeling the spread of the virus. This HMM will have 7 hidden states, called
# severity levels. Severity is derived by combining a region's monthly cases and deaths. Severity 1 is the lowest,
# 7 the highest.
# 
# For the observations, we collected the number of new cases and deaths for the regions of Munich and Landkreis Munich (where Garching
# is located).
# 
# A purely fictional Bavarian king, named Markus IV.,
# rolled his seven-sided die and has it proclaimed that our exam can only take place on-site if the severity
# level in February 2022 is below 6 in Munich and Garching. Will he allow us to conduct the exam in person?
# 
# Your task is to complete the missing code. Make sure that all the functions follow the provided interfaces of the
# functions, i.e., the output of the function exactly matches the description in the docstring.
# You only have to modify code that is in a block like this:
# 
# ```
# ##########################################################
# # YOUR CODE HERE
# .....
# ##########################################################
# ```

# ### Learning Outcomes:

# * Implementation of the Viterbi algorithm
# * Implementation of smoothing and prediction

# ### Import

# For testing and grading, we want to state that you are not allowed to import
# any other libraries and should not change the structure of the provided functions
# (i.e, the arguments and the name of the functions).

# In[1]:


import numpy as np

import visualization.visualize_data as vis
import helpers.utils as ut


# ### Corona data visualization (Nothing to implement here)

# Before we start to model the corona situation, we would like to visualize the data first to have a basic understanding of it. Feel free to change some parameters to achieve different visualization settings. Have fun!

# In[2]:


# load the corona data
corona_df, munich_map_df = vis.load_data()


# In[3]:


# check what munich_map_df looks like
munich_map_df


# In[4]:


# extract Munich and LK Munich
munich_map_df = vis.extract_data(munich_map_df)


# In[5]:


# check what corona_df looks like
corona_df


# In[6]:


# Both dataframes have the column "GEN", we merge them using this column
joined_df = vis.join_dataframes(corona_df, munich_map_df)

# Check the final dataframe before we do the visualization
joined_df


# In[7]:


# visualize the daily confirmed cases
colour_map_cases, slider_map_cases = vis.visualize_corona_data(joined_df, "Confirmed")
colour_map_deaths, slider_map_deaths = vis.visualize_corona_data(joined_df, "Deaths")


# In[8]:


# final visualization, drag the bar to see the statistics on different days
#slider_map_cases
# uncomment to see the deaths
slider_map_deaths


# ### Build the HMM model

# In our HMM model, there are two observations: Confirmed cases per 100,000 people and death cases per 5,000,000 people.
# Unlike the umbrella case in the textbook, our data is continuous, which means we need a continuous HMM model.
# But don't worry! The transition from a discrete HMM model to a continuous one is easy. In this task, you are only
# supposed to implement a discrete model.
# 
# Let's define a class named HMM that is able to execute filtering, smoothing, prediction and the Viterbi algorithm
# based on discrete observation data. You will implement the aforementioned functions step by step
# within the **HMM** class. Vectorized operations given in the textbook will be very useful for during implementation.
# 
# **Note: During implementation, we refer the state with their associated indices.**
# For example, as defined in the textbook, the ij-th entry of transition matrix denotes the transformation probability
# from state j to state i.

# In[9]:


class HMM():
    def __init__(self,
                 observation_mu,
                 observation_sigma,
                 transition_matrix,
                 initial_probability):
        assert len(transition_matrix) == len(observation_mu) == len(observation_sigma)
        self.trans_mat = transition_matrix
        self.obs_mat = ut.get_obs_mat(observation_mu, observation_sigma)
        self.init_prob = initial_probability
        self.N_state = len(self.trans_mat)

    def compute_observation_matrix(self, observation):
        """ Compute the observation matrix for vectorized operation.

        Args:
            observation: observation in some timestamp

        Return:
            O_matrix: observation matrix

        """
        prob_density = [self.obs_mat[i].pdf(observation) for i in range(self.N_state)]
        O_matrix = np.diag(prob_density)
        return O_matrix


    def forward_onestep(self, f, observation):
        """ Compute one forward step for filtering.
            N stands for number of hidden states.
            Hint: Use the O_matrix we provided for one step forward opration.

        Args:
            f: numpy array with shape [N, ], vector of f_{1:t} depicting probability of state given previous observation sequence
            observation: observed state in timestamp t+1


        Return:
            f_onestep: numpy array with shape [N, ], updated vector of f_{1:t+1}

        """
        # Acquire the row vector and transform it to diagonal matrix
        O_matrix = self.compute_observation_matrix(observation)
        f_onestep = None
        #######################################
        # YOUR CODE HERE
        f_onestep = O_matrix @ self.trans_mat @ f
        #######################################
        return f_onestep


    def backward_onestep(self, b, observation):
        """ Compute one backward step for smoothing
            N stands for number of hidden states.
            Hint: Use the O_matrix we provided for one step backward.

        Args:
            b: numpy array with shape [N, ], vector of b_{k+2:T} depicting probability of observation sequence given state
            observation: observed state in timestamp k+1

        Return:
            b_onestep: numpy array with shape [N, ], updated vector of b_{k+1:T}

        """
        # Acquire the row vector and transform it to diagonal matrix
        O_matrix = self.compute_observation_matrix(observation)
        b_onestep = None
        #######################################
        # YOUR CODE HERE
        b_onestep = self.trans_mat.T @ O_matrix @ b
        #######################################
        return b_onestep


    def forward_backward(self, observation_sequence):
        """ forward-backward algorithm for smoothing
            In this function, you will finalize the forward-backward algorithm based on your previous implementations
            of function “forward_onestep” and "backward_onestep".

            Remember to normalize the result smoothed probability to ensure the probability sum to be 1!
            Note: We provide normalization function as a reference, you can also implement your own :)

            T stands for sequence length and N stands for number of hidden states.

        Args:
            observation_sequence: observed sequence in a given period with length T


        Return:
            smoothed_prob: numpy array with shape [T, N], state probability in this period after smoothing.

        """
        forward_prob = {}
        smoothed_prob = {}
        backward_prob = {}
        #######################################
        # YOUR CODE HERE

        # Initialize the parameters
        T = len(observation_sequence)
        forward_prob[0] = self.init_prob
        backward_prob[T+1] = np.ones(self.N_state)

        for i in range(0, T, 1):
            forward_prob[i+1] = self.forward_onestep(forward_prob[i], observation_sequence[i])

        for i in range(T, 0, -1):
            smoothed_prob[i] = self.normalize(forward_prob[i] * backward_prob[i+1], axis=0)
            backward_prob[i] = self.backward_onestep(backward_prob[i+1], observation_sequence[i-1])

        #######################################
        return ut.dict_to_array(smoothed_prob)


    def predict(self, observation_sequence, k):
        """ Predict state probability for future timestamp, remember to filter the given observation sequence at first.

            Remember to normalize the result forward probability to ensure the probability sum to be 1!
            Note: We provide normalization function as a reference.

            N stands for number of hidden states.

        Args:
            observation_sequence: observed sequence in a given period with length T
            k: the number of steps of timestamp T

        Return:
            p: numpy array with shape [N, ], vector of state probability after k steps of timestamp T in the future

        """
        #######################################
        # YOUR CODE HERE
        forward_prob = {}
        pred_history = {}

        # Initialize the parameters
        T = len(observation_sequence)
        forward_prob[0] = self.init_prob

        # Calculate forward variable f_{1:t}
        for i in range(0, T, 1):
            forward_prob[i+1] = self.forward_onestep(forward_prob[i], observation_sequence[i])

        # Predict
        prediction = self.normalize(forward_prob[T])
        for i in range(k):
            prediction = self.trans_mat @ prediction
            pred_history[i] = prediction

        #######################################
        return prediction


    def get_predicted_state(self, prediction):
        """ Find the index of the most likely state in the prediction.

        Args:
            prediction: numpy array with shape [N, ], vector of state probabilities

        Return:
            state: integer that is the index of the most likely state in prediction
        """
        state = {}
        ######################################################
        # YOUR CODE HERE
        state = np.where(prediction == np.amax(prediction))
        state = state[0][0]
        ########################################################
        return state


    def viterbi(self, observation_sequence):
        """ Compute the most likely state trajectory given a observation sequence.
            T stands for sequence length.

        Args:
            observation_sequence: observed sequence in a given period with length T

        Return:
            trajectory: numpy array with shape [T, ], trajectory of the most likely state in the given period.
            Note that the output trajectory is in the form of the corresponding indices of the states. For example,
            the output with numpy array [1, 7, 5] denotes given a sequence with length T=3, the most likely state
            transforms from state 1 to state 7 and finally to state 5.
        """
        max_prob = {}
        most_likely_state = {}
        trajectory = {}
        #######################################
        # YOUR CODE HERE

        T = len(observation_sequence)

        # Initialization with filtering
        start_time = 1
        max_prob[start_time] = self.forward_onestep(self.init_prob, observation_sequence[start_time-1])

        # Compute probability
        for i in range(start_time+1, T+1):

            # Acquire the probability and current observation
            prob = self.trans_mat * max_prob[i-1][np.newaxis] # N x N

            # Keep track of the most likely state
            most_likely_state[i-1] = np.argmax(prob, axis = 1)

            # Compute the maximum probability
            observation = observation_sequence[i-1]
            observation_prob = np.diagonal(self.compute_observation_matrix(observation))
            max_logit =  observation_prob * np.amax(prob, axis = 1)

            # Normalize the probability
            max_prob[i] = self.normalize(max_logit)


        # Initialize the trajectory and do back-tracking
        trajectory[T] = np.argmax(max_prob[T])

        for i in range(T-1, 0, -1):
            trajectory[i] = most_likely_state[i][trajectory[i+1]]

        #######################################
        return ut.dict_to_array(trajectory)

    @staticmethod
    def normalize(logit, axis=0):
        prob = logit / np.sum(logit, axis=axis)
        return prob


# ### Apply HMM for COVID-19 Severity Estimation

# In this section, your final task is to use the use HMM model for COVID-19 severity estimation as a guide for examination decision-makers.
# 
# You may refer to this [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9543670) to better understand how we can leverage HMM to model the spread progress of COVID-19 given some observation data such as daily and cumulative infections and daily deaths. (This paper will only appear here and not in the final version.)
# 
# In this task, we define the following components of HMM as follows:
# 
# ```
# Hidden State: Severity of the COVID, there are 7 discrete levels in total and the transition diagram is shown in below.
# Observation: A 7 * 2 matrix (mus) and a 7 * 2 * 2 matrix (sigmas)
# Transformation Model: A 7 * 7 matrix that depicts the internal relationship between each state.
# Sensor Model: A 2-dimensional gaussian distribution based on given state.
# ```
# 
# It can be a little tricky since in this scenario the sensor model is modeled as a continuous probability density distribution instead of a discrete matrix as discussed during lectures. While don't be panic since we have  provided you a helper function to extract the diagonal observation matrix (O_matrix), and then the algorithm for filtering, smoothing and viterbi will be just the same as the discrete case we dealt with during class.
# 
# We will provide pre-estimated HMM parameters for you such as transformation matrix and preprocessed data, let's now take a look on how we use HMM to model COVID-19.

# ### Transition Matrix

# We define the transitional matrix as the following figure shows:
# ![title](img/transitional_matrix.png)

# In[10]:


#######################################
# T is a 7 * 7 matrix, T(ij) means the probability from state i to state j

T = np.array([[0.989, 0.002, 0.008, 0.000, 0.000, 0.000, 0.000],
              [0.004, 0.984, 0.005, 0.004, 0.004, 0.000, 0.000],
              [0.007, 0.002, 0.977, 0.009, 0.000, 0.000, 0.000],
              [0.000, 0.007, 0.010, 0.973, 0.006, 0.000, 0.009],
              [0.000, 0.005, 0.000, 0.009, 0.977, 0.014, 0.004],
              [0.000, 0.000, 0.000, 0.000, 0.010, 0.981, 0.003],
              [0.000, 0.000, 0.000, 0.005, 0.003, 0.005, 0.984]])


# ### Observation Matrix

# We define the observational matrix as the following figure shows:
# <img src='img/observational_matrix.png'  width='35%'  height='35%'>

# In[11]:


#######################################
# mus is a 7 * 2 matrix, which indicates the input mean of gaussian variable
# sigmas is a 7 * 2 * 2 matrix, which indicates the input covariance of gaussian variable. Each state has a 2 * 2 covariance matrix

mus = np.array([[0.000, 2.842],
                [7.105, 43.526],
                [8.365, 5.425],
                [54.292, 35.294],
                [123.655, 150.556],
                [255.332, 390.887],
                [331.681, 162.652]])


sigmas = np.zeros((7, 2, 2))
sigmas[0] = np.array([[0.049, 0],
                      [0, 14.896]])
sigmas[1] = np.array([[41.209, 16.856],
                      [16.856, 664.93]])
sigmas[2] = np.array([[39.984, 11.417],
                      [11.417, 12.691]])
sigmas[3] = np.array([[658.07, 310.464],
                      [310.464, 430.269]])
sigmas[4] = np.array([[3584.35, 1716.96],
                      [1716.96, 2281.93]])
sigmas[5] = np.array([[28086.8, 17875.2],
                      [17875.2, 19007.1]])
sigmas[6] = np.array([[26891.2, 15268.4],
                      [15268.4, 14259]])


# ### Initialized State Distribution

# In[12]:


# init_prob is a 7-dimensional vector
init_prob = np.array([0.414, 0, 0.401, 0.137, 0, 0, 0.047])


# ### Observation Sequence

# Since we take into account 21 months in total, the sequence is a 21 * 2 matrix which contains both confirmed cases and death cases.
# 
# **The sequence can be found in sequence.csv**

# In[13]:


# observation sequence is a 21 * 2 matrix containing both confirmed cases and death cases, here, we define 2 sequences, one for Munich,
# one for LK_Munich

obs_seq_Munich = np.array([[183, 358],
                           [45, 54],
                           [24, 1],
                           [34, 0],
                           [110, 0],
                           [191, 14],
                           [415, 124],
                           [914, 708],
                           [969, 1282],
                           [507, 559],
                           [166, 114],
                           [371, 160],
                           [601, 208],
                           [278, 115],
                           [70, 5],
                           [96, 0],
                           [273, 73],
                           [537, 165],
                           [743, 207],
                           [1730, 462],
                           [1149, 149]])

obs_seq_LK_Munich = np.array([[178, 587],
                              [34, 18],
                              [9, 0],
                              [20, 0],
                              [58, 0],
                              [101, 0],
                              [361, 4],
                              [790, 416],
                              [767, 917],
                              [491, 590],
                              [241, 166],
                              [341, 6],
                              [550, 0],
                              [245, 0],
                              [44, 0],
                              [65, 0],
                              [191, 20],
                              [391, 22],
                              [563, 0],
                              [1734, 26],
                              [1119, 32]])


# ### Running

# In[14]:


# Run Forward-backward algorithm
hmm = HMM(mus, sigmas, T, init_prob)
smoothed_prob_Munich = hmm.forward_backward(obs_seq_Munich)
smoothed_prob_LK_Munich = hmm.forward_backward(obs_seq_LK_Munich)
print('smoothed_prob_Munich: \n', smoothed_prob_Munich, '\n')
print('smoothed_prob_LK_Munich: \n', smoothed_prob_LK_Munich, '\n')

# Run Prediction
predicted_prob_Munich = hmm.predict(obs_seq_Munich, 2)
predicted_prob_LK_Munich = hmm.predict(obs_seq_LK_Munich, 2)
print('prediction_Munich: \n', predicted_prob_Munich, '\n')
print('prediction_LK_Munich: \n', predicted_prob_LK_Munich, '\n')
print('Predicted severity state Munich: \n', hmm.get_predicted_state(predicted_prob_Munich))
print('Predicted severity state LK Munich: \n', hmm.get_predicted_state(predicted_prob_LK_Munich))

# Test viterbi algorithm
trajectory_Munich = hmm.viterbi(obs_seq_Munich)
trajectory_LK_Munich = hmm.viterbi(obs_seq_LK_Munich)
print('trajectory_Munich: \n',trajectory_Munich, '\n')
print('trajectory_LK_Munich: \n',trajectory_LK_Munich, '\n')


# In[15]:


# visualize the result in plt
months = ['Apr 2020', 'May 2020', 'Jun 2020', 'Jul 2020', 'Aug 2020', 'Sep 2020', 'Oct 2020', 'Nov 2020',
              'Dec 2020', 'Jan 2021', 'Feb 2021', 'Mar 2021', 'Apr 2021', 'May 2021', 'Jun 2021', 'Jul 2021',
              'Aug 2021', 'Sep 2021', 'Oct 2021', 'Nov 2021', 'Dec 2021']

vis.vis_plt(trajectory_Munich, 'Severity of COVID-19 in Munich', months)
vis.vis_plt(trajectory_LK_Munich, 'Severity of COVID-19 in LK-Munich', months)


# In[16]:


# visualize the result in DataFrame
data = [months, list(trajectory_Munich), list(trajectory_LK_Munich)]
df = vis.vis_dataframe(data)
df


# In[ ]:




