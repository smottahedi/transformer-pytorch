"""Unittest for add and norm layer"""

import unittest
import torch

from transformer.add_and_norm import AddNorm # pylint: disable=import-error


class TestAddNorm(unittest.TestCase):

    def setUp(self):
        self.input_a = torch.ones(2, 3, 4) # pylint: disable=no-member
        self.input_b = torch.ones(2, 3, 4) # pylint: disable=no-member
        self.add_norm = AddNorm(3, 4, 0.5)

    def test_add_and_norm(self):
        output = self.add_norm(self.input_a, self.input_b)
        self.assertEqual(output.shape, torch.Size((2, 3, 4))) # pylint: disable=no-member


if __name__ == '__main__':
    unittest.main() 