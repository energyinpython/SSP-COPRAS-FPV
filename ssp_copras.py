import itertools
import numpy as np
from pyrepo_mcda import normalizations as norms
from mcda_method import MCDA_method


class COPRAS(MCDA_method):
    def __init__(self, normalization_method = norms.sum_normalization):
        """
        Create the COPRAS method object and select normalization method `normalization_method`

        Parameters
        -----------
            normalization_method : function
                method for decision matrix normalization chosen from `normalizations`
        """
        self.normalization_method = normalization_method

    def __call__(self, matrix, weights, types, s_coeff = 0):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.
            s_coeff: ndarray
                Vector with sustainability coefficient determined for each criterion

        Returns
        --------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ----------
        >>> copras = COPRAS(normalization_method = sum_normalization)
        >>> pref = copras(matrix, weights, types, s_coeff)
        >>> rank = rank_preferences(pref, reverse = True)
        """

        COPRAS._verify_input_data(matrix, weights, types)
        return COPRAS._copras(self, matrix, weights, types, self.normalization_method, s_coeff)


    # function for applying the SSP paradigm
    def _equalization(self, matrix, types, s_coeff):

        # Calculate mean deviation multiplied by s coefficient
        mad = (matrix - np.mean(matrix, axis = 0)) * s_coeff

        # Set as 0, those mean deviation values that for profit criteria are lower than 0
        # and those mean deviation values that for cost criteria are higher than 0
        for j, i in itertools.product(range(matrix.shape[1]), range(matrix.shape[0])):
            # for profit criteria
            if types[j] == 1:
                if mad[i, j] < 0:
                    mad[i, j] = 0
            # for cost criteria
            elif types[j] == -1:
                if mad[i, j] > 0:
                    mad[i, j] = 0

        # Subtract from performance values in decision matrix standard deviation values multiplied by a sustainability coefficient.
        return matrix - mad

    @staticmethod
    def _copras(self, matrix, weights, types, normalization_method, s_coeff):
        # reducing compensation in normalized decision matrix
        e_matrix = self._equalization(matrix, types, s_coeff)
        # Normalize matrix as for profit criteria using chosen normalization method.
        # norm_matrix = matrix/np.sum(matrix, axis = 0)
        norm_matrix = normalization_method(e_matrix, np.ones(len(weights)))
        # Multiply all values in the normalized matrix by weights.
        d = norm_matrix * weights
        # Calculate the sums of weighted normalized outcomes for profit criteria.
        Sp = np.sum(d[:, types == 1], axis = 1)
        # Calculate the sums of weighted normalized outcomes for cost criteria.
        Sm = np.sum(d[:, types == -1], axis = 1)
        # Calculate the relative priority Q of evaluated options.
        Q = Sp + ((np.sum(Sm))/(Sm * np.sum(1 / Sm)))
        # Calculate the quantitive utility value for each of the evaluated options.
        U = Q / np.max(Q)
        return U