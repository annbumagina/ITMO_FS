import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from filters.VDM import VDM
from filters.FitCriterion import FitCriterion
from filters.GiniIndexFilter import GiniIndexFilter
from filters.SpearmanCorrelationFilter import SpearmanCorrelationFilter
from hybrid import Melif


def filter_test(filter, X, y, answer):
    try:
        assert filter.run(X, y).keys() == answer.keys()
        print('Test passed')
    except AssertionError:
        print('Test failed')


def wrapper_test(wrapper):
    pass  # TODO test wrapper algorithm


def electricity_preprocess():
    data = pd.read_csv('data/electricity-normalized.csv')
    data['class'] = data['class'].apply(lambda x: 0 if x == 'DOWN' else 1)
    target = data['class']
    data = data.drop(['class'], axis=1)
    return data, target


def breast_preprocess():
    data = pd.read_csv("data/matrix", sep=' ', header=None)
    target = data[data.columns[0]]
    data.drop(data.columns[-1], axis=1, inplace=True)
    data.drop(data.columns[0], axis=1, inplace=True)
    return data, target



electricity_X, electricity_y = electricity_preprocess()
breast_X, breast_y = breast_preprocess()
filter_tests = [
    (SpearmanCorrelationFilter(0.3), electricity_X, electricity_y,
     {'date': 0.9999999989252104, 'day': 0.9999999499088564, 'period': 0.9999999991477627,
      'nswprice': 0.9999999989323045, 'nswdemand': 0.9999999993656636, 'vicprice': 0.9999999987695205,
      'vicdemand': 0.9999999993251367, 'transfer': 0.999999999146249}),
    (GiniIndexFilter(0.3), electricity_X, electricity_y, {'date': 0.37209712153031527}),
    (SpearmanCorrelationFilter(0.3), breast_X, breast_y, {}),
    # (FitCriterion(), breast_X, breast_y, {}),
    # (VDM(), breast_X, breast_y, {})
]


def cut(x, border):
    if type(x) is dict:
        return dict([i for i in x.items() if i[1] > border])
    if type(x) is np.ndarray or type(x) is list:
        return [i for i in x if i > border]


if __name__ == '__main__':
    for test in filter_tests:
        filter_test(*test)

    estimator = RandomForestClassifier()

    wrapper = Melif([SpearmanCorrelationFilter(-1), GiniIndexFilter(-1)])
    wrapper.fit(breast_X, breast_y)
    wrapper.run(cut, estimator)