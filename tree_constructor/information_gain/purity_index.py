import numpy as np

class PurityIndex():
    """A class for calculating the purity of a categorical data column"""

    def __init__(self, index_type="entropy"):
        self.__set_purity_index(index_type)

    def __set_purity_index(self, index_type):
        if index_type == "entropy":
            self.calculate = self.__entropy 
        elif index_type == "gini":
            self.calculate = self.__gini

    def __entropy_term(self, probability):
        return -probability * np.log2(probability) 

    def __gini_term(self, probability):
        return probability * (1 - probability)

    def __calculate(self, values, index_term):
        purity = 0.0 
        for value in np.unique(values):
            probability = self.probability_of_value(value, values)
            purity += index_term(probability)
        return float(purity)

    def __entropy(self, values):
        return self.__calculate(values, self.__entropy_term)

    def __gini(self, values):
        return self.__calculate(values, self.__gini_term) 

    def probability_of_value(self, value, values):
        """Calculates the probability of seeing the given value in a column of categorical values"""
        return float( np.count_nonzero(values == value) / len(values) )
