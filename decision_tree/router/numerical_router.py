import numpy as np

from decision_tree.feature_thresholds import numerical_threshold
from decision_tree.router.router import Router

class NumericalRouter(Router):
    """A class for routing data based on numerical (continous) features"""

    def __init__(self, feature_col, feature_col_values):
        super().__init__(feature_col, feature_col_values)
        self.__threshold = numerical_threshold(feature_col_values)

    def route(self, datapoint, from_node):
        feature_value = datapoint[self._feature_col]
        if feature_value <= self.__threshold:
            return from_node.left_child()
        else:
            return from_node.right_child()

    def feature_has_more_than_two_categories(self):
        return False

    def _row_mask(self, feature_values):
        return feature_values > self.__threshold 
