from typing import Sequence
import numpy as np
import numpy.typing as npt
import permutation_utils


class PermutationEnvironment:

    state: Sequence[int]
    permutations_added_array: npt.NDArray[bool]
    last_added_array = npt.NDArray[bool]
    def __init__(self, alphabet: Sequence[int]):
        self.alphabet = alphabet
        self.possible_permutations = permutation_utils.get_all_permutations(alphabet)
        self.number_of_possible_permutations = len(self.possible_permutations)
        self.permutations_added_array = np.zeros(self.number_of_possible_permutations, dtype=bool)
        self.last_added_array = np.zeros(self.number_of_possible_permutations, dtype=bool)
        self.state = []

    def add_permutation(self, index: int):
        permutation_to_add = list(self.possible_permutations[index])
        self.last_added_array = np.zeros(self.number_of_possible_permutations, dtype=bool)
        self.last_added_array[index] = True
        if len(self.state) == 0:
            self.state = list(permutation_to_add)
        else:
            self.state = permutation_utils.merge_permutations(self.state, permutation_to_add)
        # String of two merged permutations may contain another permutation
        new_ids = permutation_utils.get_permutation_ids_contained_in_symbols_string(self.state[-2*len(self.alphabet):], self.alphabet)
        self.permutations_added_array[new_ids] = True

    def get_observation(self):
        return np.concatenate([self.permutations_added_array, self.last_added_array])