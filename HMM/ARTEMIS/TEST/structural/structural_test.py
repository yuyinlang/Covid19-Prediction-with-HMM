import unittest
import sys, os


#from timeout_decorator import *
import structural_helpers
import assignment.HMM as HMM

try:
    import assignment.HMM as HMM
    importFlag = True
except:
    importFlag = False
    
print(importFlag)


class TestStructural(unittest.TestCase):
    TIMEOUT_CONSTANT = 180
    time_error = "Importing the notebook took more than {TIMEOUT_CONSTANT} seconds. This is longer than expected. Please make sure that every cell is compiling and prevent complex structures."
    import_error = "There seems to be an error in the provided notebook. Please make sure that every cell is compiling without an error."
    method_error = "Function %s could not be found. Please don\'t rename the methods."
    library_error = "You can only use the given libraries, please check if you imported other libraries."

    #@timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=time_error)
    #def test_notebook_import(self):
    #    if (importFlag is False):
    #        raise ImportError(self.import_error)
    #    else:
     #       pass

    #def test_imported_librabries(self):
        #self.assertIs(structural_helpers.check_imported_libraries(HMM), True, self.library_error)

    #def test_check_function_names(self):
        #self.assertIs(structural_helpers.check_for_function('forward_onestep', HMM), True, self.method_error % ('forward_onestep'))
        #self.assertIs(structural_helpers.check_for_function('backward_onestep', HMM), True, self.method_error % ('backward_onestep'))
        #self.assertIs(structural_helpers.check_for_function('forward_backward', HMM), True, self.method_error % ('forward_backward'))
        #self.assertIs(structural_helpers.check_for_function('predict', HMM), True, self.method_error % ('predict'))
        #self.assertIs(structural_helpers.check_for_function('viterbi', HMM), True, self.method_error % ('viterbi'))
        #self.assertIs(structural_helpers.check_for_function('get_predicted_state', HMM), True, self.method_error % ('get_predicted_state'))


# To test it locally
#if __name__ == '__main__':
#    unittest.main()

