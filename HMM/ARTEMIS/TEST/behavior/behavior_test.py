import unittest

from timeout_decorator import *
import HMM_solution as sol
import numpy as np
import sys, os


try:
    import assignment.HMM as HMM

    importFlag = True
except:
    importFlag = False


class TestNotebook(unittest.TestCase):
    TIMEOUT_CONSTANT = 240
    TIME_ERROR = 'Timeout Error. %s seems to take more than 4m to execute. Please keep the computational complexity in mind.'

    def setUp(self):

        self.T1 = T = np.array([[0.2, 0.6],
                                [0.8, 0.4]])
        self.mus1 = np.array([[0.8, 0.7],
                              [0.3, 0.2]])
        self.sigmas1 = np.zeros((2, 2, 2))
        self.sigmas1[0] = np.eye(2)
        self.sigmas1[1] = np.eye(2)
        self.init_prob1 = np.array([0.3, 0.7])
        self.hmm1_sol = sol.HMM(self.mus1, self.sigmas1, self.T1, self.init_prob1)
        self.hmm1 = HMM.HMM(self.mus1, self.sigmas1, self.T1, self.init_prob1)

        self.prediction1 = np.array([2, 6, 9, 2, 66, 9, 0])

        self.T2 = np.array([[0.989, 0.002, 0.008, 0.000, 0.000, 0.000, 0.000],
                            [0.004, 0.984, 0.005, 0.004, 0.004, 0.000, 0.000],
                            [0.007, 0.002, 0.977, 0.009, 0.000, 0.000, 0.000],
                            [0.000, 0.007, 0.010, 0.973, 0.006, 0.000, 0.009],
                            [0.000, 0.005, 0.000, 0.009, 0.977, 0.014, 0.004],
                            [0.000, 0.000, 0.000, 0.000, 0.010, 0.981, 0.003],
                            [0.000, 0.000, 0.000, 0.005, 0.003, 0.005, 0.984]])

        self.mus2 = np.array([[0.000, 2.842],
                              [7.105, 43.526],
                              [8.365, 5.425],
                              [54.292, 35.294],
                              [123.655, 150.556],
                              [255.332, 390.887],
                              [331.681, 162.652]])

        self.sigmas2 = np.zeros((7, 2, 2))
        self.sigmas2[0] = np.array([[0.049, 0],
                                    [0, 14.896]])
        self.sigmas2[1] = np.array([[41.209, 16.856],
                                    [16.856, 664.93]])
        self.sigmas2[2] = np.array([[39.984, 11.417],
                                    [11.417, 12.691]])
        self.sigmas2[3] = np.array([[658.07, 310.464],
                                    [310.464, 430.269]])
        self.sigmas2[4] = np.array([[3584.35, 1716.96],
                                    [1716.96, 2281.93]])
        self.sigmas2[5] = np.array([[28086.8, 17875.2],
                                    [17875.2, 19007.1]])
        self.sigmas2[6] = np.array([[26891.2, 15268.4],
                                    [15268.4, 14259]])
        self.init_prob2 = np.array([0.414, 0, 0.401, 0.137, 0, 0, 0.047])
        self.hmm2_sol = sol.HMM(self.mus2, self.sigmas2, self.T2, self.init_prob2)
        self.hmm2 = HMM.HMM(self.mus2, self.sigmas2, self.T2, self.init_prob2)
        self.prediction2 = np.array([3.66, 8.933, 75.55, 222.222, 335.4])

        obs_seq1 = np.array([[0.9, 0.2], [7.0, 1.2], [11.0, 0.01], [0.33, 0.45], [0.72, 0.33]])
        obs_seq2 = np.array([[0.1, 0.02], [0.12, 0.3], [9.1, 0.1], [0.93, 0.15], [1.72, 0.03]])
        obs_seq3 = np.array([[0.1, 0.1], [0.02, 0.8], [1.8, 1.8], [0.0, 0.15], [1.0, 0.03]])
        self.obs_seqs = [obs_seq1, obs_seq2, obs_seq3]


    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError,
                               exception_message=TIME_ERROR % ('forward_backward()'))
    def test_forward_backward(self):
        a = self.hmm1.forward_backward(self.mus1)
        b = self.hmm1_sol.forward_backward(self.mus1)
        try:
            self.assertEqual(True, np.array_equal(a, b))
        except:
            raise AssertionError("Error in forward_backward")

        try:
            self.assertEqual(a.shape, b.shape)
        except:
            raise AssertionError("Array has wrong shape")

        a = self.hmm2.forward_backward(self.mus2)
        b = self.hmm2_sol.forward_backward(self.mus2)
        try:
            self.assertEqual(True, np.array_equal(a, b))
        except:
            raise AssertionError("Error in forward_backward")
        try:
            self.assertEqual(a.shape, b.shape)
        except:
            raise AssertionError("Array has wrong shape")


    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError,
                               exception_message=TIME_ERROR % ('predict()'))
    def test_prediction(self):
        a = self.hmm1.predict(self.mus1, 5)
        b = self.hmm1_sol.predict(self.mus1, 5)
        try:
            self.assertEqual(True, np.array_equal(a, b))
        except:
            raise AssertionError("Error in prediction")

        a = self.hmm2.predict(self.mus2, 5)
        b = self.hmm2_sol.predict(self.mus2, 5)
        try:
            self.assertEqual(True, np.array_equal(a, b))
        except:
            raise AssertionError("Error in prediction")

    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError,
                               exception_message=TIME_ERROR % ('predict()'))
    def test_viterbi(self):
        a = self.hmm1.viterbi(self.mus1)
        b = self.hmm1_sol.viterbi(self.mus1)
        try:
            self.assertEqual(True, np.array_equal(a, b))
        except:
            raise AssertionError("Error in viterbi")
        try:
            self.assertEqual(a.shape, b.shape)
        except:
            raise AssertionError("Array has wrong shape")

        a = self.hmm2.viterbi(self.mus2)
        b = self.hmm2_sol.viterbi(self.mus2)
        try:
            self.assertEqual(True, np.array_equal(a, b))
        except:
            raise AssertionError("Error in viterbi")
        try:
            self.assertEqual(a.shape, b.shape)
        except:
            raise AssertionError("Array has wrong shape")

    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError,
                               exception_message=TIME_ERROR % ('get_state()'))
    def test_get_state(self):
        a = self.hmm1.get_predicted_state(self.prediction1)
        b = self.hmm1_sol.get_predicted_state(self.prediction1)
        try:
            self.assertEqual(True, np.array_equal(a, b))
        except:
            raise AssertionError("Wrong state returned")

# To test it locally
# if __name__ == '__main__':
#    unittest.main()
