import numpy as np

from decision_tree.feature_thresholds import numerical_threshold
from tree_constructor.information_gain.purity_index import PurityIndex

class NumericalPurityIndex(PurityIndex):

    def _calculate(self, values, index_term):
        threhold = numerical_threshold(values)
        return float( - self.probability_of_value(threhold - 1, values) - self.probability_of_value(threhold + 1, values) )

    def probability_of_value(self, value, values):
        threshold = numerical_threshold(values)
        binarized_on_threshold = values > threshold 
        if value <= threshold:
            return float( (len(binarized_on_threshold) - np.count_nonzero(binarized_on_threshold)) / len(binarized_on_threshold) )
        else:
            return float( np.count_nonzero(binarized_on_threshold) / len(binarized_on_threshold) )
        
