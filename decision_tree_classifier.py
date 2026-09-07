import numpy as np
from sklearn.base import BaseEstimator

from tree_constructor.data_handler import TrainingData
from tree_constructor.tree_grower import TreeGrower
from decision_tree.decision_tree import DecisionTree


class DecisionTreeClassifier(BaseEstimator):

    def __init__(self, criterion="entropy", max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.__tree = DecisionTree() 
        self.__is_fit = False

    def __check_criterion(self):
        if self.criterion not in ["entropy", "gini"]:
            raise ValueError("The classifier criterion must be either 'entropy' or 'gini'")

    def fit(self, X_train, y_train):
        data = TrainingData.create_from_pandas_data(X_train, y_train)
        tree_grower = TreeGrower(self.__tree, criterion=self.criterion, max_depth=self.max_depth)
        tree_grower.grow_tree(data) 
        # tree_pruner = TreePruner(self.__tree)
        # tree_pruner.prune_tree()
        self.__is_fit = True

    def predict(self, X):
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
