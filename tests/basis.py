import unittest
import torch
import math
import numpy as np

from model import BernLayer

class TestBernLayer(unittest.TestCase):
    def setUp(self):
        self.degrees = 4
        self.t_values = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float32)
        
    def expected_basis_functions(self, t, degrees):
        t = float(t)
        indices = np.arange(degrees + 1)
        binom_coeff = np.array([math.comb(degrees, i) for i in indices])
        basis = np.zeros(degrees + 1)
        
        for i in indices:
            basis[i] = binom_coeff[i] * (t ** i) * ((1 - t) ** (degrees - i))
        
        return basis

    def test_get_basis(self):
        for t in self.t_values:
            basis = BernLayer.get_basis(t.unsqueeze(0), self.degrees).numpy().flatten()
            expected_basis = self.expected_basis_functions(t, self.degrees)
            np.testing.assert_almost_equal(basis, expected_basis, decimal=6,
                                           err_msg=f"Basis functions for t={t.item()} are incorrect.")
            
if __name__ == "__main__":
    unittest.main()
