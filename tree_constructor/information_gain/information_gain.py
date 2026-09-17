import numpy as np

from decision_tree.feature_thresholds import numerical_threshold
from tree_constructor.information_gain.purity_index import PurityIndex

class InformationGain:

    def __init__(self, data, purity_index="entropy"):
        self.__data = data
        self.__purity = PurityIndex(index_type=purity_index) 

    def __purity_given_categorical_feature(self, feature_values, labels):
        purity = 0.0
        unique_feature_values = np.unique(feature_values)
        for i in range(len(unique_feature_values)):
            feature_value = unique_feature_values[i]
            probability_of_feature_value = self.__purity.probability_of_value(feature_value, feature_values)
            labels_given_feature_value = labels[feature_values == feature_value]
            purity += probability_of_feature_value * self.__purity.calculate(labels_given_feature_value)
        return float(purity)

    def __purity_given_numerical_feature(self, feature_values, labels):
        purity = 0.0
        threshold = numerical_threshold(feature_values)
        labels_of_smaller_feature_values = labels[feature_values <= threshold]
        labels_of_larger_feature_values = labels[feature_values > threshold]
        probability_of_smaller_feature_value = len(labels_of_smaller_feature_values) / len(labels)
        probability_of_larger_feature_value = len(labels_of_larger_feature_values) / len(labels)
        purity += probability_of_smaller_feature_value * self.__purity.calculate(labels_of_smaller_feature_values)
        purity += probability_of_larger_feature_value * self.__purity.calculate(labels_of_larger_feature_values)
        return purity

    def __information_gain_given_categorical_feature(self, feature_values, labels):
        return self.__purity.calculate(labels) - self.__purity_given_categorical_feature(feature_values, labels)

    def __information_gain_given_numerical_feature(self, feature_values, labels):
        return self.__purity.calculate(labels) - self.__purity_given_numerical_feature(feature_values, labels)

    def feature_with_highest_information_gain(self):
        best_col = 0
        highest_information_gain = 0
        labels = self.__data.labels()
        for feature_col in range(self.__data.num_features()):
            feature_values = self.__data.feature_values(feature_col)
            if self.__data.is_numerical_feature(feature_col):
                information_gain = self.__information_gain_given_numerical_feature(feature_values, labels)
            else:
                information_gain = self.__information_gain_given_categorical_feature(feature_values, labels)
            if information_gain > highest_information_gain:
                best_col = feature_col
                highest_information_gain = information_gain
        return best_col
