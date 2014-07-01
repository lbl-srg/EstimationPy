'''
Created on Feb 25, 2014

@author: marco
'''
import unittest
from ukf.ukfFMU import ukfFMU

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_InstantiateUKF(self):
        filter = ukfFMU()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()