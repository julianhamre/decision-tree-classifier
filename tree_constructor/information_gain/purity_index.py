from abc import ABC, abstractmethod
import numpy as np

class PurityIndex(ABC):

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

    @abstractmethod
    def _calculate(self, values, index_term) -> float:
        pass

    def __entropy(self, values):
        return self._calculate(values, self.__entropy_term)

    def __gini(self, values):
        return self._calculate(values, self.__gini_term) 

    @abstractmethod
    def probability_of_value(self, value, values) -> float:
        pass
