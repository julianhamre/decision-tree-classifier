import numpy as np

class TrainingData:

    @staticmethod
    def create_from_pandas_data(X, y):
        return TrainingData(X.to_numpy(), y.to_numpy(), TrainingData.numerical_feature_colums(X))
    
    @staticmethod
    def numerical_feature_colums(X):
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
        if self.is_numerical_feature(feature_col):
            return self.__X[:, feature_col]
        else:
            return np.round(self.__X[:, feature_col]).astype(int)

    def is_numerical_feature(self, feature_col):
        return feature_col in self.__numerical_features

    def __concatenated_matrix(self):
        y = self.__y.reshape(-1, 1)
        return np.concatenate((self.__X, y), axis=1)

    def divided_data(self, feature_col, row_mask):
        matrix = self.__concatenated_matrix()
        left_child_matrix = matrix[~row_mask(self.feature_values(feature_col))]
        right_child_matrix = matrix[row_mask(self.feature_values(feature_col))]
        left_child_data = TrainingData(left_child_matrix[:, :-1], left_child_matrix[:, -1], self.__numerical_features)
        right_child_data = TrainingData(right_child_matrix[:, :-1], right_child_matrix[:, -1], self.__numerical_features)
        return left_child_data, right_child_data
        

