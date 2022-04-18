import os
import unittest
import pickle

from code.ml import Model

model = Model()

file_path = os.path.dirname(os.path.realpath(__file__)) + '/../output_boundary_revised.txt'

class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(file_path, "w"):
            pass

    def test_is_model_underfitting(self):
        try:
            X_train, X_test, y_train, y_test = model.data_transformation()
            predictions = model.model_predict(X_test)

            benchmark_msle = model.cost_metric(
                y_true=y_test, y_pred=[y_train.mean()]*y_test.shape[0]
            )

            predicted_msle = model.cost_metric(
                y_true=y_test, y_pred=predictions
            )

            if predicted_msle < benchmark_msle:
                passed = True
                with open(file_path, "a") as f:
                    f.write("TestModelNotUnderfitting=True\n")
                    print("TestModelNotUnderfitting = Passed")
            else:
                passed = False
                with open(file_path, "a") as f:
                    f.write("TestModelNotUnderfitting=False\n")
                    print("TestModelNotUnderfitting = Failed")
        except:
            passed = False
            with open(file_path, "a") as f:
                f.write("TestModelNotUnderfitting=False\n")
                print("TestModelNotUnderfitting = Failed")

        assert passed

    def test_is_model_overfitting(self):

        try:
            X_train, X_test, y_train, y_test = model.data_transformation()

            train_predict = model.model_predict(X_train)
            train_msle = model.cost_metric(
                y_true=y_train.values, y_pred=train_predict
            )

            test_predict = model.model_predict(X_test)        
            test_msle = model.cost_metric(
                y_true=y_test.values, y_pred=test_predict
            )

            perc_10 = (train_msle/100)*10

            diff = abs(train_msle-test_msle)

            if diff < perc_10:
                passed = True
                with open(file_path, "a") as f:
                    f.write("TestModelNotOverfitting=True\n")
                    print("TestModelNotOverfitting = Passed")
            else:
                passed = False
                with open(file_path, "a") as f:
                    f.write("TestModelNotOverfitting=False\n")
                    print("TestModelNotOverfitting = Failed")
        except:
            passed = False
            with open(file_path, "a") as f:
                f.write("TestModelNotOverfitting=False\n")
                print("TestModelNotOverfitting = Failed")
        assert passed
