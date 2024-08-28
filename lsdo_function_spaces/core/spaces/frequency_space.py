import numpy as np
import scipy.sparse as sps
from lsdo_function_spaces import FunctionSpace, Function
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Union
import csdl_alpha as csdl


class FrequencySpace(FunctionSpace):
    def __init__(self, num_parametric_dimensions:int, order:tuple[int]):
        """
        Frequency space - fit via fourrier transform.

        Parameters
        ----------
        num_parametric_dimensions : int
            The number of parametric dimensions.
        frequencies : np.ndarray
            The frequencies of the fourrier series. shape should be (num_frequencies, num_parametric_dimensions) 
        """
        if isinstance(order, int):
            order = (order, ) * num_parametric_dimensions
        self.order = order

        super().__init__(num_parametric_dimensions, (np.prod(order),))

    def compute_basis_matrix(self, parametric_coordinates:np.ndarray, parametric_derivative_orders:np.ndarray=None, expansion_factor:int=None) -> np.ndarray:
        """
        Compute the basis matrix for the given parametric coordinates.
        """
        # if parametric_derivative_orders is not None:
        #     raise NotImplementedError('FrequencySpace does not support derivatives')
        weights = np.zeros((parametric_coordinates.shape[0], np.prod(self.order))*2+1)
        start = 1
        for i, order in enumerate(self.order):
            end = start + 2*order
            weights[:, start:end:2] = np.sin(2*np.pi*np.arange(1, order+1)*parametric_coordinates[:,i])
            weights[:, start+1:end+1:2] = np.cos(2*np.pi*np.arange(1, order+1)*parametric_coordinates[:,i])
            start = end

        # add a column of ones for the constant term
        weights[:, 0] = 1
        return weights
    
    def generate_coefficient_vector(self, constant, sin_coefficients:csdl.Variable, cos_coefficients:csdl.Variable) -> csdl.Variable:
        """
        Generate the coefficient vector for the given sin and cos coefficients.
        """

        coefficients = np.zeros(np.prod(self.order)*2+1)
        coefficients = csdl.Variable(value=coefficients)
        start = 0
        for i, order in enumerate(self.order):
            end = start + 2*order
            coefficients = coefficients.set(csdl.slice[start:end:2], sin_coefficients[i])
            coefficients = coefficients.set(csdl.slice[start+1:end+1:2], cos_coefficients[i])
        coefficients[0] = constant
        return coefficients

# def test_polynomial_space():
#     num_parametric_dimensions = 2
#     order = 2
#     space = PolynomialSpace(num_parametric_dimensions=num_parametric_dimensions, order=order)
#     parametric_coordinates = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
#     basis_matrix = space.compute_basis_matrix(parametric_coordinates)
#     assert basis_matrix.shape == (3, 9)
#     assert np.allclose(basis_matrix, [[1., 0.1, 0.01, 0.2, 0.02, 0.01, 0.04, 0.008, 0.004],
#                                       [1., 0.3, 0.09, 0.4, 0.12, 0.04, 0.16, 0.048, 0.016],
#                                       [1., 0.5, 0.25, 0.6, 0.3, 0.15, 0.36, 0.18, 0.09]])

#     order = (2, 3)
#     space = PolynomialSpace(num_parametric_dimensions=num_parametric_dimensions, order=order)
#     parametric_coordinates = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
#     basis_matrix = space.compute_basis_matrix(parametric_coordinates)
#     assert basis_matrix.shape == (3, 18)
#     assert np.allclose(basis_matrix, [[1., 0.1, 0.01, 0.2, 0.02, 0.01, 0.04, 0.008, 0.004, 0.008, 0.0016, 0.0008, 0.016, 0.0032, 0.0016, 0.032, 0.0064, 0.0032],
#                                       [1., 0.3, 0.09, 0.4, 0.12, 0.04, 0.16, 0.048, 0.016, 0.024, 0.0072, 0.0024, 0.064, 0.0192, 0, 0.128, 0.0384, 0.0128],
#                                         [1., 0.5, 0.25, 0.6, 0.3, 0.15, 0.36, 0.18, 0.09, 0.04, 0.02, 0.01, 0.1, 0.05, 0.025, 0.2, 0.1, 0.05]])
    

# if __name__ == '__main__':
#     test_polynomial_space()
#     print('PolynomialSpace tests passed.')