import numpy as np
from sklearn.base import BaseEstimator

from tree_constructor.data_handler import TrainingData
from tree_constructor.tree_grower import TreeGrower
from decision_tree.decision_tree import DecisionTree


class DecisionTreeClassifier(BaseEstimator): # Extends the Sklearn BaseEstimator to enable the cross_val_score function to clone the model 

    def __init__(self, criterion="entropy", max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.__tree = DecisionTree() 
        self.__is_fit = False

    def __check_criterion(self):
        if self.criterion not in ["entropy", "gini"]:
            raise ValueError("The classifier criterion must be either 'entropy' or 'gini'")

    def fit(self, X_train, y_train):
        """
        Fits the decision tree classifier on the given data.

        Args:
             X_train (pandas.DataFrame): A dataframe containing all feature columns. Categorical features must be represented as integers, and numerical features as floats.
             y_train (pandas.Series): A series of integer label values
        """
        data = TrainingData.create_from_pandas_data(X_train, y_train)
        tree_grower = TreeGrower(self.__tree, criterion=self.criterion, max_depth=self.max_depth)
        tree_grower.grow_tree(data) 
        self.__is_fit = True

    def predict(self, X):
        """
        Predicts the label values of the given X-values. 
        The X-values must have the same number of columns and contain the same datatypes as the X-values fitted on.

        Args:
             X (pandas.DataFrame): A dataframe containing all feature columns. Categorical features must be represented as integers, and numerical features as floats.

        Returns:
            numpy.array: The predicted labels
        """
        if not self.__is_fit:
            raise RuntimeError("The model must be fitted before predicting")
        y = [] 
        first_node = self.__tree.root()
        for datapoint in X.itertuples(index=False):
            y.append(self.__decision(datapoint, first_node))
        return np.array(y)
            
    def __decision(self, datapoint, parent_node):
        if parent_node.is_leaf():
            return parent_node.decision()
        child_node = parent_node.route(datapoint)
        return self.__decision(datapoint, child_node)
