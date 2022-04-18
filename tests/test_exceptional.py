import os
import unittest

from code.ml import Model
from code import constants

model = Model()

file_path = os.path.dirname(os.path.realpath(__file__)) + '/../output_exception_revised.txt'

class ExceptionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(file_path, "w"):
            pass

    def test_model_exists(self):

        try:
            model_exists = os.path.exists(constants.MODEL_FOLDER + constants.MODEL_FILE)

            if model_exists:
                passed  = True
                with open(file_path, "a") as f:
                    f.write("TestModelCreated = True\n")
                    print("TestModelCreated = Passed")
            else:
                passed = False
                with open(file_path, "a") as f:
                    f.write("TestModelCreated = False\n")
                    print("TestModelCreated = Failed")
        except:
            passed = False
            with open(file_path, "a") as f:
                f.write("TestModelCreated = False\n")
                print("TestModelCreated = Failed")

        assert passed

    def test_graphs_exists(self):

        try:
            n_graphs = len(os.listdir(constants.GRAPHS_FOLDER))

            if n_graphs != 0:
                passed = True
                with open(file_path, "a") as f:
                    f.write("TestGraphsGenerated = True\n")
                    print("TestGraphsGenerated = Passed")
            else:
                passed = False
                with open(file_path, "a") as f:
                    f.write("TestGraphsGenerated = False\n")
                    print("TestGraphsGenerated = Failed")
        except:
            passed = False
            with open(file_path, "a") as f:
                f.write("TestGraphsGenerated = False\n")
                print("TestGraphsGenerated = Failed")

        assert passed
