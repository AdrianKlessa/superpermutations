import unittest
import permutation_utils
import numpy as np


class TestPermutationUtils(unittest.TestCase):
    def test_get_permutations(self):
        alphabet = [1, 2, 3]
        expected = [
            (1, 2, 3),
            (1, 3, 2),
            (2, 1, 3),
            (2, 3, 1),
            (3, 1, 2),
            (3, 2, 1)
        ]
        actual = permutation_utils.get_all_permutations(alphabet)
        self.assertEqual(expected, actual)

    def test_is_superpermutation(self):
        alphabet1 = [1, 2]
        perm1 = np.array([1, 2])
        perm2 = np.array([2, 1])
        perm3 = np.array([1, 2, 2, 1])
        perm4 = np.array([1, 2, 1])
        self.assertEqual(False, permutation_utils.is_superpermutation(perm1, alphabet1))
        self.assertEqual(False, permutation_utils.is_superpermutation(perm2, alphabet1))
        self.assertEqual(True, permutation_utils.is_superpermutation(perm3, alphabet1))
        self.assertEqual(True, permutation_utils.is_superpermutation(perm4, alphabet1))

        alphabet2 = [1, 2, 3, 4, 5]
        perm5 = np.array([int(i) for i in
                          "123451234152341253412354123145231425314235142315423124531243512431524312543121345213425134215342135421324513241532413524132541321453214352143251432154321"])
        self.assertEqual(True, permutation_utils.is_superpermutation(perm5, alphabet2))
        perm6 = np.array([int(i) for i in
                          "12345123415234125341235412314523142531423514231542312453124351243152431254312134521342513421534213542132451324153241352413254132145321435214325143215432"])
        self.assertEqual(False, permutation_utils.is_superpermutation(perm6, alphabet2))

    def test_get_overlap(self):
        seq1 = [1, 2, 3]
        seq2 = [3, 2, 1]

        actual_overlap = permutation_utils.get_permutation_overlap(seq1, seq2)
        self.assertEqual(1, actual_overlap)

        seq1 = [1, 2, 3]
        seq2 = [2, 3, 1]

        actual_overlap = permutation_utils.get_permutation_overlap(seq1, seq2)
        self.assertEqual(2, actual_overlap)

        seq1 = [1, 2, 3, 4, 5]
        seq2 = [3, 4, 5, 1, 2]

        actual_overlap = permutation_utils.get_permutation_overlap(seq1, seq2)
        self.assertEqual(3, actual_overlap)

    def test_merge_permutations(self):
        perm1 = [1, 2, 3]
        perm2 = [3, 2, 1]
        merged = [1,2,3,2,1]
        self.assertEqual(merged, permutation_utils.merge_permutations(perm1, perm2))

        perm1 = [1, 2]
        perm2 = [2, 1]
        merged = [1,2,1]
        self.assertEqual(merged, permutation_utils.merge_permutations(perm1, perm2))