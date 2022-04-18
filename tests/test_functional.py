import os
import unittest
import pickle

from code.ml import Model

model = Model()
file_path = file_path = os.path.dirname(os.path.realpath(__file__)) + '/../output_revised.txt'

try:
    X_train, X_test, y_train, y_test = model.data_transformation()
except:
    pass

class FuctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(file_path, "w"):
            pass

    def test_train_test_split_reproduciability(self):
        try:
            X_train2, X_test2, y_train2, y_test2 = model.data_transformation()

            if (X_test.equals(X_test2)):
                passed = True
                with open(file_path, "a") as f:
                    f.write("TestTrainTestSplitReproduciability=True\n")
                    print("TestTrainTestSplitReproduciability = Passed")
            else:
                passed = False
                with open(file_path, "a") as f:
                    f.write("TestTrainTestSplitReproduciability=False\n")
                    print("TestTrainTestSplitReproduciability = Failed")
        except:
            passed = False
            with open(file_path, "a") as f:
                f.write("TestTrainTestSplitReproduciability=False\n")
                print("TestTrainTestSplitReproduciability = Failed")
        assert passed

    def test_train_vs_test_features(self):

        try:
            if (X_test.shape[1] == X_train.shape[1]):
                passed = True
                with open(file_path, "a") as f:
                    f.write("TestTrainTestFearturesCardinality=True\n")
                    print("TestTrainTestFearturesCardinality = Passed")
            else:
                passed = False
                with open(file_path, "a") as f:
                    f.write("TestTrainTestFearturesCardinality=False\n")
                    print("TestTrainTestFearturesCardinality = Failed")
        except:
            passed = False
            with open(file_path, "a") as f:
                f.write("TestTrainTestFearturesCardinality=False\n")
                print("TestTrainTestFearturesCardinality = Failed")
        assert passed

    def test_numerical_features_imputed(self):
        try:
            na_found_train = X_train.isnull().any().any()
            na_found_test = X_test.isnull().any().any()

            if not (na_found_train or na_found_test):
                passed = True
                with open(file_path, "a") as f:
                    f.write("TestFeaturesNANotFound=True\n")
                    print("TestFeaturesNANotFound = Passed")
            else:
                passed = False
                with open(file_path, "a") as f:
                    f.write("TestNumericalFeaturesNANotFound=False\n")
                    print("TestNumericalFeaturesNANotFound = Failed")
        except:
            passed = False
            with open(file_path, "a") as f:
                f.write("TestNumericalFeaturesNANotFound=False\n")
                print("TestNumericalFeaturesNANotFound = Failed")
        assert passed

    def test_pipeline_custom_model_random_state(self):
        try:
            loaded_model = pickle.load(open(model.model_file, 'rb'))
            model_random_state = loaded_model.random_state

            if model_random_state is not None:
                passed = True
                with open(file_path, "a") as f:
                    f.write("TestIsModelRandSeedSet=True\n")
                    print("TestIsModelRandSeedSet = Passed")
            else:
                passed = False
                with open(file_path, "a") as f:
                    f.write("TestIsModelRandSeedSet=False\n")
                    print("TestIsModelRandSeedSet = Failed")
        except:
            passed = False
            with open(file_path, "a") as f:
                    f.write("TestIsModelRandSeedSet=False\n")
                    print("TestIsModelRandSeedSet = Failed")
        assert passed
        
