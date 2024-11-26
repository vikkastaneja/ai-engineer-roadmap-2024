import sys

sys.path.append('./')
# sys.path.append('Tutorial\ 4\ -\ Gradient\ Descent')
import numpy as np
import matplotlib.pyplot as plt
from gradient import gradient_descent_algorithm as gd
import unittest
from sklearn import linear_model
import pandas as pd
import pytest

@pytest.fixture
def testdata():
    return [np.array([[1,2,3,4,5],[5,7,9,11,13]]),
            np.array([[92,56,88,70,80,49,65,35,66,67],[98,68,81,80,83,52,66,30,68,73]])]

def test_gradient_descent(testdata):
    for data in testdata:
        print('-----> ', type(data))
        print('------', data)
        actual = gd(data[0], data[1])
        model = linear_model.LinearRegression()
        df = pd.DataFrame(data.T, columns=["X","Y"])
        model.fit(df.drop('Y', axis=1), df.drop('X', axis=1))
        print('====', float(model.coef_[0][0]), float(actual[0]))
        print('++++', float(model.intercept_[0]), float(actual[1]))
        np.testing.assert_allclose(round(model.coef_[0][0], 2), round(actual[0], 2))
        np.testing.assert_allclose(round(model.intercept_[0], 2), round(actual[1], 2))
