from decision_tree.node import Node

class DecisionTree:

    def __init__(self):
        self.__root = Node()

    def root(self):
        return self.__root

    def assign_left_child(self, parent_node, child_node):
        parent_node.assign_left_child(child_node)
        child_node.assign_parent(parent_node) 

    def assign_right_child(self, parent_node, child_node):
        parent_node.assign_right_child(child_node)
        child_node.assign_parent(parent_node)


