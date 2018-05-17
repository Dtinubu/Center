import unittest

import torch
import numpy

from loss import get_center_delta, compute_center_loss

class TestCenterLossFunctions(unittest.TestCase):

    def setUp(self):
        # Mock features, centers and targets
        self.features = torch.tensor( (( 1,2,3), (4,5,6), (7,8,9)) ).float()
        self.centers = torch.tensor( ((1,1,1), (2,2,2), (3,3,3), (5,5,5) )).float()
        self.targets = torch.tensor((1, 3, 1))
        self.lamda = 0.5

    def test_get_center_delta(self):
        result = get_center_delta(self.features, self.centers, self.targets)

        # size should match
        self.assertTrue(result.size() == self.centers.size())

        # for class 1
        class1_result = ((self.features[0] + self.features[2]) - 2 * self.centers[1]) / 3
        self.assertEqual(3, torch.sum(result[1] == class1_result).item())

        # for class 3
        class3_result = (self.features[1] - self.centers[3]) / 2
        self.assertEqual(3, torch.sum(result[3] == class3_result).item())

        # others should all be zero
        sum_others = torch.sum(result[(0,2), :]).item()
        self.assertEqual(0, sum_others)

    def test_compute_center_loss(self):
        class1_loss = torch.sum(torch.pow( self.features[(0,2), :] - self.centers[(1,1), :], 2 ))
        class3_loss = torch.sum(torch.pow( self.features[1] - self.centers[(3), :], 2 ))
        total_loss = self.lamda / 2 * (class1_loss + class3_loss).item()
        self.assertEqual(total_loss, compute_center_loss(self.features, self.centers, self.targets, self.lamda))

if __name__ == '__main__':
    unittest.main()