import numpy as np

class TrainingData:

    @staticmethod
    def create_from_pandas_data(X, y):
        return TrainingData(X.to_numpy(), y.to_numpy(), TrainingData.numerical_feature_colums(X))
    
    @staticmethod
    def numerical_feature_colums(X):
        """
        Retrieves the column indices that contain a numerical (continous) variable.
        These are the columns containing float values.

        Args:
            X (pandas.DataFrame): A dataframe containing all feature values. Categorical features must be integers, and numerical features must be floats.
        Returns:
            list: A list of column indices
        """
        column_indices = []
        for col in X.select_dtypes(include=[ "float" ]).columns:
            column_indices.append(X.columns.get_loc(col))
        return column_indices

    def __init__(self, X, y, numerical_features):
        self.__numerical_features = numerical_features
        self.__X = X
        self.__y = y.flatten()

    def is_empty(self):
        return len(self.__X) == 0 or len(self.__y) == 0

    def labels(self):
        """Retrieves the integer label values of the data"""
        return np.round(self.__y).astype(int)

    def labels_are_equal(self):
        if len(self.labels()) <= 1:
            return True
        return np.all(self.labels() == self.labels()[0])

    def features_are_equal(self):
        if len(self.__X) <= 1:
            return True
        for col in range(self.__X.shape[1]):
            feature_values = self.feature_values(col)
            if not np.allclose(feature_values, feature_values[0], atol=1e-5):
                return False
        return True

    def num_features(self):
        return self.__X.shape[1]

    def feature_values(self, feature_col):
        """
        Retrieves the column values of the given feature col index.
        If the feature is categorical, the returned array will contain integer values.
        Otherwise, it will contain floats."""
        if self.is_numerical_feature(feature_col):
            return self.__X[:, feature_col]
        else:
            return np.round(self.__X[:, feature_col]).astype(int)

    def is_numerical_feature(self, feature_col):
        """Returns True if the feature in the column given by the feature_col index is a numerical (continous) feature"""
        return feature_col in self.__numerical_features

    def __concatenated_matrix(self):
        y = self.__y.reshape(-1, 1)
        return np.concatenate((self.__X, y), axis=1)

    def divided_data(self, feature_col, row_mask):
        """
        Divids the data in two based on the given row masking function.
        The two data partitions are meant for continued tree growth from two child nodes of a common parent node.

        Args:
            feature_col (int): The index of the feature column to divide based on. 
            row_mask (function): A function on the values of the feature column index. It constructs the row mask boolean array by filtering this column.
        Returns:
            tuple: A tuple containing two TrainingData objects. 
            The first object contains the rows aligning with the row_masks False values. 
            The second object contains the rows aligning with the row_masks True values.
            The first and second data objects are ment for the left and right child nodes respectively.
        """
        matrix = self.__concatenated_matrix()
        left_child_matrix = matrix[~row_mask(self.feature_values(feature_col))]
        right_child_matrix = matrix[row_mask(self.feature_values(feature_col))]
        left_child_data = TrainingData(left_child_matrix[:, :-1], left_child_matrix[:, -1], self.__numerical_features)
        right_child_data = TrainingData(right_child_matrix[:, :-1], right_child_matrix[:, -1], self.__numerical_features)
        return left_child_data, right_child_data
        

