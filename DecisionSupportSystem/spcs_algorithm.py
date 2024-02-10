import math
import numpy as np
from typing import Optional, List, Tuple
from operator import itemgetter


class SPCS:
    """
    Class for calculating SPCS algorithm.
    """
    def __init__(self, alt: np.ndarray, min_max: np.ndarray, metric: str) -> None:
        # Getting inputs:
        self.alt = {i + 1: alt[i] for i in range(len(alt))}
        self.min_max = min_max
        self.metric = metric

        # Initializing needed fields:
        self.matrix: List[List[float]] = []
        self.norm_matrix: List[List[float]] = []
        self.quo_point: List[float] = []
        self.asp_point: List[float] = []
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.sc_val: List[Tuple[int, float]] = []

    def __delete_dominated_points(self) -> None:
        """
        Deleting dominated points.
        """
        to_delete = []
        for point in self.alt.keys():
            if point not in to_delete and self.__check_dominated(point):
                to_delete.append(point)
        for point in to_delete:
            del self.alt[point]

    def __check_dominated(self, alt_nr: int) -> bool:
        """
        Checking if a point is dominated.
        """
        # Creating handle for min/max comparisons to check if elem is dominated:
        def compare(to_compare: float, comparing_to: float, min_max: bool) -> bool:
            # We maximize:
            if min_max:
                return to_compare <= comparing_to
            # We minimalize:
            return to_compare >= comparing_to

        # Going over all alternatives:
        is_dominated = False
        for point in self.alt:
            # Avoid checking alternative with itself:
            if point != alt_nr:
                # Checking whether a point is dominated by looking through its coordinates
                # and applying corresponding min_max flags:
                count = 0
                for idx in range(len(self.alt[alt_nr])):
                    if compare(self.alt[alt_nr][idx], self.alt[point][idx], self.min_max[idx]):
                        count += 1
                # If both coordinates are dominated:
                if count == len(self.alt[alt_nr]):
                    is_dominated = True
                    break
        return is_dominated

    def __create_matrix(self) -> None:
        """
        Creating alternatives matrix.
        """
        for point, values in self.alt.items():
            self.matrix.append([point, values[0], values[1]])

    def __normalize_alt(self) -> None:
        """
        Normalizing alternatives.
        """
        if self.alt:
            max_param1 = -1 * float("inf")
            max_param2 = -1 * float("inf")
            for values in self.alt.values():
                max_param1 = values[0] if values[0] > max_param1 else max_param1
                max_param2 = values[1] if values[1] > max_param2 else max_param2
            for point in self.alt:
                self.alt[point][0] /= max_param1
                self.alt[point][1] /= max_param2

    def __create_normalized_matrix(self) -> None:
        """
        Normalizing created matrix.
        """
        for point, values in self.alt.items():
            self.norm_matrix.append([point, values[0], values[1]])

    def __set_quo_point(self) -> None:
        """
        Calculating quo points - average values of criteria.
        """
        first_param = 0
        second_param = 0
        n = len(self.alt)
        for point in self.alt:
            first_param += self.alt[point][0]
            second_param += self.alt[point][1]
        self.quo_point = [round(first_param / n, 2), round(second_param / n, 2)]

    def __set_aspiration_point(self) -> None:
        """
        Calculating aspiration points - best possible combination of criteria values.
        """
        # Worst values of parameters for minimization - the highest ones:
        first_param = float("inf")
        second_param = float("inf")

        # If we encounter True, that means that we want to maximize - the worst value is the smallest one:
        if self.min_max[0]:
            first_param *= -1
        if self.min_max[1]:
            second_param *= -1

        for point in self.alt:
            first_param = max(self.alt[point][0], first_param) if self.min_max[0] \
                else min(self.alt[point][0], first_param)
            second_param = max(self.alt[point][1], second_param) if self.min_max[1] \
                else min(self.alt[point][1], second_param)
        self.asp_point = [first_param, second_param]

    def __count_scoring_values(self) -> None:
        """
        Calculating scoring values of alternatives.
        """
        # Dict of scoring values related to alternatives:
        sc_values = []

        # Count slope of line between quo_point and asp_point:
        a = (self.asp_point[1] - self.quo_point[1]) / (self.asp_point[0] - self.quo_point[0])
        b = self.quo_point[1] - a * self.quo_point[0]
        self.a = a
        self.b = b

        line_length = float('inf')

        # Check selected metric:
        if self.metric == "Euclidean":
            # Getting length of line - L2 distance:
            line_length = math.sqrt((self.quo_point[1] - self.asp_point[1]) ** 2 + (self.quo_point[0] - self.asp_point[0]) ** 2)
        elif self.metric == "Chebyshev":
            # Getting length of line - Chebyshev distance:
            line_length = np.max(np.abs(np.array(self.quo_point) - np.array(self.asp_point)))

        for point, params in self.alt.items():
            # Coordinates of alternatives:
            x, y = params[0], params[1]

            # Distance of alternative to line:
            d = abs(-a * x + y - b) / math.sqrt(a ** 2 + 1)

            # Getting line perpendicular to quo-asp line and going through alt point:
            a_new = -1 / a
            b_new = y - a_new * x

            # Getting intersection of 2 lines - point:
            A = np.array([[a, -1],
                          [a_new, -1]])
            B = np.array([-b, -b_new])

            x_perp, y_perp = np.linalg.solve(A, B)

            dist = float('inf')

            # Check selected metric:
            if self.metric == "Euclidean":
                # Getting distance between perp_point and quo point (beginning of the section) - L2 distance:
                dist = math.sqrt((self.quo_point[1] - y_perp) ** 2 + (self.quo_point[0] - x_perp) ** 2)
            elif self.metric == "Chebyshev":
                # Getting distance between perp_point and quo point (beginning of the section) - Chebyshev distance:
                dist = np.max(np.abs(np.array(self.quo_point) - np.array([x_perp, y_perp])))

            # Parametrizing:
            p = dist / line_length

            # If asp point is on the right and quo point on the left:
            if self.asp_point[0] > self.quo_point[0]:
                # If x of perp point is on the right of asp point:
                if x_perp > self.asp_point[0]:
                    # We give it negative value since then the sum of p + dist will be smaller
                    # - the alternative is better than aspiration point:
                    p *= -1
            # If asp point is on the left and quo point on the right:
            else:
                # If x of perp point is on the left of asp point:
                if x_perp < self.asp_point[0]:
                    # We give it negative value since then the sum of p + dist will be smaller
                    # - the alternative is better than aspiration point:
                    p *= -1

            # Getting scoring values:
            sc_values.append((point, d + p))

        # We sort ascending - the one with the smallest scoring is the best (distance from asp point):
        self.sc_val = sorted(sc_values, key=itemgetter(1))

    def calc(self) -> None:
        """
        Calculating the algorithm.
        """
        self.__delete_dominated_points()
        self.__create_matrix()
        self.__normalize_alt()
        self.__create_normalized_matrix()
        self.__set_quo_point()
        self.__set_aspiration_point()
        self.__count_scoring_values()
