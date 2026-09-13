import numpy as np

from tree_constructor.information_gain.information_gain import InformationGain 
from decision_tree.node import Node

class TreeGrower:

    def __init__(self, tree, criterion="entropy", max_depth=None):
        self.__tree = tree
        self.__criterion = criterion
        self.__max_depth = max_depth

    def grow_tree(self, data):
        """Fits the decision tree based on the given data.

        Args:
            data (TrainingData) A training data object
        """
        root_node = self.__tree.root()
        self.__grow_from(root_node, data)

    def __grow_from(self, node, data):
        """Runs the ID3 algorithm recursively from the given node"""
        if data.labels_are_equal() or data.features_are_equal():
            self.__complete_leaf(node, data)
            return
        information_gain = InformationGain(data, purity_index=self.__criterion)
        best_feature_col = information_gain.feature_with_highest_information_gain()
        self.__complete_subtree(node, best_feature_col, data)

    def __node_is_at_max_depth(self, node):
        if self.__max_depth is None:
            return False
        return node.depth() >= self.__max_depth
         
    def __complete_leaf(self, node, data):
        self.__assign_decision(node, data)

    def __complete_subtree(self, subtree_root, feature_col, data):
        """
        Completes the subtree of the given node by assigning its decision, routing
        and branching by splitting the data on the given feature column.
        The node will become a leaf if the maximum tree depth is reached.
        """
        self.__add_node_content(subtree_root, feature_col, data)
        if self.__node_is_at_max_depth(subtree_root):
            return
        self.__branch_from(subtree_root, data)   

    def __add_node_content(self, node, feature_col, data):
        self.__train_router(node, feature_col, data)
        self.__assign_decision(node, data)

    def __train_router(self, node, feature_col, data):
        numerical_feature = data.is_numerical_feature(feature_col)
        if numerical_feature:
            node.train_numerical_router(feature_col, data.feature_values(feature_col))
        else:
            node.train_categorical_router(feature_col, data.feature_values(feature_col))
 
    def __assign_decision(self, node, data):
        decision = np.bincount(data.labels()).argmax()
        node.assign_decision(decision)

    def __branch_from(self, node, data):
        router = node.router()
        left_child_data, right_child_data = router.divided_data(data)
        if left_child_data.is_empty() or right_child_data.is_empty():
            return
        left_child = Node()
        right_child = Node()
        self.__tree.assign_left_child(node, left_child)
        self.__tree.assign_right_child(node, right_child)

        # This is a crucial step. 
        # If the feature to split on is categorical, and has more than two categories, 
        # the tree should have been split using a multi-branch (non-binary) split.
        # Since my tree is a binary tree, i achieve the multi-branch split
        # by a sequence of binary splits. In other words, I force
        # the algorithm to split again on the same feature until 
        # the feature no longer contains multiple categories in one data partition.
        if router.feature_has_more_than_two_categories():
            self.__complete_subtree(left_child, router.feature_col(), left_child_data)
        else:
            self.__grow_from(left_child, left_child_data)
        self.__grow_from(right_child, right_child_data)
  
