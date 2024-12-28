from typing import Sequence
import numpy as np
import numpy.typing as npt
import itertools
import permutation_utils


class PermutationEnvironment:

    state: Sequence[int] # The permutation as a string
    permutations_added_array: npt.NDArray[bool]
    def __init__(self, alphabet: Sequence[int]):
        self.alphabet = alphabet
        self.possible_permutations = permutation_utils.get_all_permutations(alphabet)
        self.number_of_possible_permutations = len(self.possible_permutations)
        self.permutations_added_array = np.zeros(self.number_of_possible_permutations, dtype=bool)

    def add_permutation(self, index: int):
        permutation_to_add = self.possible_permutations[index]
        self.permutations_added_array[index] = True
        self.state = permutation_utils.merge_permutations(self.state, permutation_to_add)