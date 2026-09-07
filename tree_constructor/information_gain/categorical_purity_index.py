import numpy as np

from tree_constructor.information_gain.purity_index import PurityIndex

class CategoricalPurityIndex(PurityIndex):

    def _calculate(self, values, index_term):
        purity = 0.0 
        for value in np.unique(values):
            probability = self.probability_of_value(value, values)
            purity += index_term(probability)
        return float(purity)
    
    def probability_of_value(self, value, values):
        return float( np.count_nonzero(values == value) / len(values) )
