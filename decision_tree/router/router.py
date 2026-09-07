from abc import ABC, abstractmethod
import numpy as np

def is_numerical_feature(feature_col_values):
    return np.issubdtype(feature_col_values.dtype, np.floating)

class Router(ABC):

    def __init__(self, feature_col, feature_col_values):
        self._feature_col = feature_col
        self._feature_col_values = feature_col_values

    def feature_col(self):
        return self._feature_col

    @abstractmethod
    def route(self, datapoint, from_node):
        pass

    @abstractmethod
    def feature_has_more_than_two_categories(self) -> bool:
        pass

    @abstractmethod
    def _row_mask(self, feature_values) -> np.typing.NDArray[np.bool_]:
        pass

    def divided_data(self, data):
        return data.divided_data(self._feature_col, self._row_mask)
