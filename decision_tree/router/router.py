from abc import ABC, abstractmethod
import numpy as np

class Router(ABC):
    """
    A class for routing data based on the given feature column.
    Routing means directing data to the correct child node.
    """

    def __init__(self, feature_col, feature_col_values):
        self._feature_col = feature_col
        self._feature_col_values = feature_col_values

    def feature_col(self):
        return self._feature_col

    @abstractmethod
    def route(self, datapoint, from_node):
        """
        Gives the next node in the path through the decision tree of a datapoint to be predicted.

        Args:
            datapoint (pandas.Series): A row of feature values
            from_node (Node): The parent node to be routed from

        Returns:
            The child node to traverse to
        """
        pass

    @abstractmethod
    def feature_has_more_than_two_categories(self) -> bool:
        pass

    @abstractmethod
    def _row_mask(self, feature_values) -> np.typing.NDArray[np.bool_]:
        """
        Retrieves the row mask to divide the training data based on.
        The row mask is derived by filtering the given feature values.
        """
        pass

    def divided_data(self, data):
        """
        Divides the data in two, represented by a tuple of two TrainingData objects.
        The first tuple element is meant for a left child node to continue growing on,
        and the right element for a right child.
        """
        return data.divided_data(self._feature_col, self._row_mask)
