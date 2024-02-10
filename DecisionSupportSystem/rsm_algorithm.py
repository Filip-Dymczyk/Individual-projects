import math
import numpy as np
from operator import itemgetter
from random import uniform


class RSM:
    """
    Class for calculating RSM algorithm.
    """
    def __init__(self, alt: np.ndarray, min_max: np.ndarray, metric: str) -> None:
        # Getting inputs:
        self.alt = {i + 1: alt[i] for i in range(len(alt))}
        self.min_max = min_max
        self.metric = metric

        # Initializing fields which will be used for calculations and display:
        self.pareto_alt = None
        self.pareto_norm_alt = None
        self.lim_criteria = None
        self.pareto_alt_matrix = None
        self.alt_norm_matrix = None
        self.pareto_norm_matrix = None
        self.alt_matrix = None
        self.asp_points = None
        self.quo_points = None
        self.anti_asp_points = None
        self.optimum_lim_opt_points = None
        self.norm_asp_points = None
        self.norm_quo_points = None
        self.norm_anti_asp_points = None
        self.norm_optimum_lim_opt_points = None
        self.sc_val = None

    def __delete_dominated_points(self) -> None:
        """
        Deleting dominated points.
        """
        to_delete = []
        self.pareto_alt = dict()
        for point in self.alt:
            if point not in to_delete and self.__check_dominated(point):
                to_delete.append(point)
        for point in self.alt:
            if point not in to_delete:
                self.pareto_alt[point] = self.alt[point]

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

    def __lim_criteria_values(self) -> None:
        """
        Getting max values in each criterion.
        """
        if self.alt:
            lim_param1 = -1 * float("inf")
            lim_param2 = -1 * float("inf")
            lim_param3 = -1 * float("inf")
            for values in self.alt.values():
                lim_param1 = values[0] if values[0] > lim_param1 else lim_param1
                lim_param2 = values[1] if values[1] > lim_param2 else lim_param2
                lim_param3 = values[2] if values[2] > lim_param3 else lim_param3
            self.lim_criteria = [lim_param1 + 0.1 * lim_param1, lim_param2 + 0.1 * lim_param2,
                                 lim_param3 + 0.1 * lim_param3]

    def __norm_pareto_alt(self) -> None:
        """
        Normalizing pareto set of alternatives.
        """
        if self.pareto_alt:
            self.pareto_norm_alt = dict()
            for point in self.pareto_alt:
                x, y, z = self.pareto_alt[point]
                x /= self.lim_criteria[0]
                y /= self.lim_criteria[1]
                z /= self.lim_criteria[2]

                self.pareto_norm_alt[point] = [x, y, z]

    def __get_matrices(self) -> None:
        """
        Creating matrices from alternatives sets.
        """
        self.alt_matrix = []
        self.pareto_alt_matrix = []
        for point, values in self.alt.items():
            self.alt_matrix.append([point, values[0], values[1], values[2]])
        for point, values in self.pareto_alt.items():
            self.pareto_alt_matrix.append([point, values[0], values[1], values[2]])

    def __norm_matrices(self) -> None:
        """
        Normalizing created matrices.
        """
        self.alt_norm_matrix = []
        self.pareto_norm_matrix = []

        for point_alt, point_pareto in zip(self.alt_matrix, self.pareto_alt_matrix):
            norm_x = point_alt[1] / self.lim_criteria[0]
            norm_y = point_alt[2] / self.lim_criteria[1]
            norm_z = point_alt[3] / self.lim_criteria[2]
            self.alt_norm_matrix.append([point_alt[0], norm_x, norm_y, norm_z])

            norm_x = point_pareto[1] / self.lim_criteria[0]
            norm_y = point_pareto[2] / self.lim_criteria[1]
            norm_z = point_pareto[3] / self.lim_criteria[2]
            self.pareto_norm_matrix.append([point_pareto[0], norm_x, norm_y, norm_z])

    def __set_asp_points(self) -> None:
        """
        Calculating aspiration points - best possible combination of criteria values.
        """
        # Worst values of parameters for minimization - the highest ones:
        first_param = float("inf")
        second_param = float("inf")
        third_param = float('inf')

        # If we encounter True, that means that we want to maximize - the worst value is the smallest one:
        if self.min_max[0]:
            first_param *= -1
        if self.min_max[1]:
            second_param *= -1
        if self.min_max[2]:
            third_param *= -1

        diff_in_first = self.lim_criteria[0] * 0.05 + uniform(0, self.lim_criteria[0] * 0.05)
        diff_in_second = self.lim_criteria[1] * 0.05 + uniform(0, self.lim_criteria[1] * 0.05)

        for point in self.alt:
            first_param = max(self.alt[point][0], first_param) if self.min_max[0] \
                else min(self.alt[point][0], first_param)
            second_param = max(self.alt[point][1], second_param) if self.min_max[1] \
                else min(self.alt[point][1], second_param)
            third_param = max(self.alt[point][2], third_param) if self.min_max[2] \
                else min(self.alt[point][2], third_param)

        first_param_left = max(first_param - diff_in_first, 0)
        second_param_left = max(second_param - diff_in_second, 0)
        left_val = [first_param_left, second_param_left, third_param]

        middle_val = [first_param, second_param, third_param]

        first_param_right = first_param + diff_in_first
        second_param_right = second_param + diff_in_second
        right_val = [first_param_right, second_param_right, third_param]

        self.asp_points = [left_val, middle_val, right_val]

    def __set_quo_points(self) -> None:
        """
        Calculating quo points - average values of criteria.
        """
        first_param = 0
        second_param = 0
        third_param = 0
        diff_in_first = self.lim_criteria[0] * 0.05 + uniform(0, self.lim_criteria[0] * 0.05)
        diff_in_second = self.lim_criteria[1] * 0.05 + uniform(0, self.lim_criteria[1] * 0.05)
        n = len(self.alt)

        for point in self.alt:
            first_param += self.alt[point][0]
            second_param += self.alt[point][1]
            third_param += self.alt[point][2]

        first_param = round(first_param / n, 2)
        second_param = round(second_param / n, 2)
        third_param = round(third_param / n, 2)

        first_param_left = max(first_param - diff_in_first, 0)
        second_param_left = max(second_param - diff_in_second, 0)
        left_val = [first_param_left, second_param_left, third_param]

        middle_val = [first_param, second_param, third_param]

        first_param_right = first_param + diff_in_first
        second_param_right = second_param + diff_in_second
        right_val = [first_param_right, second_param_right, third_param]

        self.quo_points = [left_val, middle_val, right_val]

    def __set_anti_asp_points(self) -> None:
        """
        Calculating anti-aspiration points - worst possible combination of criteria values.
        """
        first_param = float("-inf")
        second_param = float("-inf")
        third_param = float("-inf")

        # If we encounter True, that means that we want to maximize - the best value is the largest one:
        if self.min_max[0]:
            first_param *= -1
        if self.min_max[1]:
            second_param *= -1
        if self.min_max[2]:
            third_param *= -1

        diff_in_first = self.lim_criteria[0] * 0.05 + uniform(0, self.lim_criteria[0] * 0.05)
        diff_in_second = self.lim_criteria[1] * 0.05 + uniform(0, self.lim_criteria[1] * 0.05)

        for point in self.alt:
            first_param = min(self.alt[point][0], first_param) if self.min_max[0] \
                else max(self.alt[point][0], first_param)
            second_param = min(self.alt[point][1], second_param) if self.min_max[1] \
                else max(self.alt[point][1], second_param)
            third_param = min(self.alt[point][2], third_param) if self.min_max[2] \
                else max(self.alt[point][2], third_param)

        first_param_left = max(first_param - diff_in_first, 0)
        second_param_left = max(second_param - diff_in_second, 0)
        left_val = [first_param_left, second_param_left, third_param]

        middle_val = [first_param, second_param, third_param]

        first_param_right = first_param + diff_in_first
        second_param_right = second_param + diff_in_second
        right_val = [first_param_right, second_param_right, third_param]

        self.anti_asp_points = [left_val, middle_val, right_val]

    def __set_optimum_lim_min(self) -> None:
        """
        Calculating optimum lim points.
        """
        # Worst values of parameters for minimization - the highest ones:
        first_param = float("inf")
        second_param = float("inf")
        third_param = float('inf')

        # If we encounter True, that means that we want to maximize - the worst value is the smallest one:
        if self.min_max[0]:
            first_param *= -1
        if self.min_max[1]:
            second_param *= -1
        if self.min_max[2]:
            third_param *= -1

        diff_in_first = self.lim_criteria[0] * 0.05 + uniform(0, self.lim_criteria[0] * 0.05)
        diff_in_second = self.lim_criteria[1] * 0.05 + uniform(0, self.lim_criteria[1] * 0.05)

        for point in self.alt:
            first_param = max(self.alt[point][0], first_param) if self.min_max[0] \
                else min(self.alt[point][0], first_param)
            second_param = max(self.alt[point][1], second_param) if self.min_max[1] \
                else min(self.alt[point][1], second_param)
            third_param = max(self.alt[point][2], third_param) if self.min_max[2] \
                else min(self.alt[point][2], third_param)

        first_param_left = max(first_param - diff_in_first, 0)
        second_param_left = max(second_param - diff_in_second, 0)
        left_val = [first_param_left, second_param_left, third_param]

        first_param_right = first_param + diff_in_first
        second_param_right = second_param + diff_in_second
        right_val = [first_param_right, second_param_right, third_param]

        self.optimum_lim_opt_points = [left_val, right_val]

    def __norm_reference_points(self) -> None:
        """
        Normalizing reference points.
        """
        lim_param1 = self.lim_criteria[0]
        lim_param2 = self.lim_criteria[1]
        lim_param3 = self.lim_criteria[2]
        self.norm_asp_points = []
        self.norm_quo_points = []
        self.norm_anti_asp_points = []
        self.norm_optimum_lim_opt_points = []

        for point_asp, point_quo, point_anti_asp in zip(self.asp_points, self.quo_points, self.anti_asp_points):
            self.norm_asp_points.append([point_asp[0] / lim_param1, point_asp[1] / lim_param2, point_asp[2] / lim_param3])
            self.norm_quo_points.append([point_quo[0] / lim_param1, point_quo[1] / lim_param2, point_quo[2] / lim_param3])
            self.norm_anti_asp_points.append([point_anti_asp[0] / lim_param1, point_anti_asp[1] / lim_param2, point_anti_asp[2] / lim_param3])

        for point in self.optimum_lim_opt_points:
            self.norm_optimum_lim_opt_points.append(
                [point[0] / lim_param1, point[1] / lim_param2, point[2] / lim_param3])

    def __get_sc_val(self) -> None:
        """
        Calculating scoring values of alternatives.
        """
        min_point_to_asp_dist = dict()
        min_point_to_lim_opt_dist = dict()
        sc_val = []
        for point in self.pareto_norm_alt:
            x, y, z = self.pareto_norm_alt[point]
            min_point_to_asp_dist[point] = float("inf")
            min_point_to_lim_opt_dist[point] = float("inf")
            dist = float("inf")

            for asp_point in self.norm_asp_points:
                asp_x, asp_y, asp_z = asp_point

                if self.metric == "Euclidean":
                    # L2 distance:
                    dist = math.sqrt((x - asp_x) ** 2 + (y - asp_y) ** 2 + (z - asp_z) ** 2)
                elif self.metric == "Chebyshev":
                    # Chebyshev distance:
                    dist = np.max(np.abs(np.array([x, y, z]) - np.array([asp_x, asp_y, asp_z])))

                if min_point_to_asp_dist[point] > dist:
                    min_point_to_asp_dist[point] = dist

            for lim_opt_points in self.norm_anti_asp_points:
                lim_opt_x, lim_opt_y, lim_opt_z = lim_opt_points

                if self.metric == "Euclidean":
                    # L2 distance:
                    dist = math.sqrt((x - lim_opt_x) ** 2 + (y - lim_opt_y) ** 2 + (z - lim_opt_z) ** 2)
                elif self.metric == "Chebyshev":
                    # Chebyshev distance:
                    dist = np.max(np.abs(np.array([x, y, z]) - np.array([lim_opt_x, lim_opt_y, lim_opt_z])))

                if min_point_to_lim_opt_dist[point] > dist:
                    min_point_to_lim_opt_dist[point] = dist
            sc_val.append((point, min_point_to_asp_dist[point] - min_point_to_lim_opt_dist[point]))

        sc_val = sorted(sc_val, key=itemgetter(1))
        self.sc_val = sc_val

    def calc(self) -> None:
        """
        Calculating the algorithm.
        """
        self.__delete_dominated_points()
        self.__lim_criteria_values()
        self.__norm_pareto_alt()
        self.__get_matrices()
        self.__norm_matrices()
        self.__set_asp_points()
        self.__set_quo_points()
        self.__set_anti_asp_points()
        self.__set_optimum_lim_min()
        self.__norm_reference_points()
        self.__get_sc_val()
