import numpy as np


class Topsis:
    """
    Class for calculating Topsis algorithm.
    """
    def __init__(self, matrix: np.ndarray[float], weight_matrix: np.ndarray[float],
                 min_max: np.ndarray[bool], metric: str) -> None:

        # M×N matrix:
        self.matrix = np.array(matrix, dtype="float64")

        # M alternatives (options):
        self.row_size = len(self.matrix)

        # N attributes/criteria:
        self.column_size = len(self.matrix[0])

        # N size weight matrix:
        self.weights = np.array(weight_matrix, dtype="float64")
        self.weights = self.weights / np.sum(self.weights)

        # Min/max criteria:
        self.min_max = min_max

        # Metric used for calculations:
        self.metric = metric

        # Initializing needed fields:
        self.normalized_decision = None
        self.weighted_normalized = None
        self.worst_alternatives = None
        self.best_alternatives = None
        self.worst_distance = None
        self.best_distance = None
        self.worst_similarity = None
        self.best_similarity = None

    def __normalize_weight(self) -> None:
        """
        Normalize criteria and calculate weight matrix.
        """
        self.normalized_decision = self.matrix / np.sqrt(np.sum(self.matrix ** 2, axis=0))

        self.weighted_normalized = self.normalized_decision * self.weights

    def __ideal_worst_cases(self) -> None:
        """
        Calculate ideal and non ideal points.
        """
        self.worst_alternatives = np.zeros(self.column_size)
        self.best_alternatives = np.zeros(self.column_size)
        for i, elem in enumerate(self.min_max):
            if elem:
                self.worst_alternatives[i] = np.min(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = np.max(
                    self.weighted_normalized[:, i])
            else:
                self.worst_alternatives[i] = np.max(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = np.min(
                    self.weighted_normalized[:, i])

    def __calculate_distance(self) -> None:
        """
        Calculate distance to best and worst point.
        """
        # Choosing selected metric:
        if self.metric == "Euclidean":
            # L2 distance:
            self.best_distance = np.sqrt(np.sum((self.weighted_normalized - self.best_alternatives) ** 2, axis=1))
            self.worst_distance = np.sqrt(np.sum((self.weighted_normalized - self.worst_alternatives) ** 2, axis=1))
        elif self.metric == "Chebyshev":
            # Chebyshev distance:
            self.best_distance = np.max(np.abs(self.weighted_normalized - self.best_alternatives), axis=1)
            self.worst_distance = np.max(np.abs(self.weighted_normalized - self.worst_alternatives), axis=1)

    def __calculate_scoring(self) -> None:
        """
        Calculating scoring values for alternatives.
        """
        np.seterr(all='ignore')
        self.worst_similarity = np.zeros(self.row_size)
        self.best_similarity = np.zeros(self.row_size)

        for i in range(self.row_size):
            # Calculate similarity to the worst condition:
            self.worst_similarity[i] = self.worst_distance[i] / \
                                       (self.worst_distance[i] + self.best_distance[i])

            # Calculate similarity to the best condition:
            self.best_similarity[i] = self.best_distance[i] / \
                                      (self.worst_distance[i] + self.best_distance[i])

    def calc(self) -> None:
        """
        Calculating the algorithm.
        """
        self.__normalize_weight()
        self.__ideal_worst_cases()
        self.__calculate_distance()
        self.__calculate_scoring()

    def rank_to_worst_similarity(self) -> list:
        """
        Generate ranking - similarity to non-ideal point.
        """
        return self.__ranking(self.worst_similarity)

    def rank_to_best_similarity(self) -> list:
        """
        Generate ranking - similarity to ideal point.
        """
        return self.__ranking(self.best_similarity)

    def __ranking(self, data) -> list:
        """
        Helper to generate appropriate ranking.
        """
        return [i + 1 for i in data.argsort()]



