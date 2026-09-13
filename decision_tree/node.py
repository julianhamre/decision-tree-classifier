from decision_tree.router.categorical_router import CategoricalRouter
from decision_tree.router.numerical_router import NumericalRouter

class Node:

    def __init__(self):
        self.__parent = None
        self.__left = None
        self.__right = None
        self.__depth = 1

    def decision(self):
        return self.__decision

    def router(self):
        return self.__router 

    def route(self, datapoint):
        """
        Routes a datapoint to be predicted by 
        returning the next node in its path 
        through the decision tree.
        """
        return self.__router.route(datapoint, self)

    def assign_decision(self, decision):
        """
        Assigns the given decision to the node.
        A decision is a label value representing
        the prediction the node will make if it
        becomes a leaf.
        """
        self.__decision = decision

    def train_numerical_router(self, feature_col, feature_values):
        self.__router = NumericalRouter(feature_col, feature_values)

    def train_categorical_router(self, feature_col, feature_values):
        self.__router = CategoricalRouter(feature_col, feature_values)

    def left_child(self):
        return self.__left

    def right_child(self):
        return self.__right

    def assign_parent(self, parent_node):
        self.__parent = parent_node 
        self.__depth = parent_node.depth() + 1

    def assign_right_child(self, right_child):
        self.__right = right_child

    def assign_left_child(self, left_child):
        self.__left = left_child

    def is_leaf(self):
        return self.__left is None and self.__right is None

    def depth(self):
        return self.__depth
