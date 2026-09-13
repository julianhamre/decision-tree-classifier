import numpy as np

from decision_tree.feature_thresholds import categorical_threshold
from decision_tree.router.router import Router

class CategoricalRouter(Router):
    """
    A class for routing data based on categorical features.
    """

    def __init__(self, feature_col, feature_col_values):
        super().__init__(feature_col, feature_col_values)
        self.__threshold = categorical_threshold(self._feature_col_values)

    def route(self, datapoint, from_node):
        feature_value = datapoint[self._feature_col]
        if feature_value == self.__threshold:
            return from_node.right_child()
        else:
            return from_node.left_child()

    def feature_has_more_than_two_categories(self):
        return len(np.unique(self._feature_col_values)) > 2

    def _row_mask(self, feature_values):
        return feature_values == self.__threshold
