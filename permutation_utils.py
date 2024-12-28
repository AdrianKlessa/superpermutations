from typing import Sequence

import numpy as np
import numpy.typing as npt
import itertools


def is_superpermutation(arr: npt.NDArray[np.int_], alphabet: Sequence[int]) -> bool:
    possible_permutations = get_all_permutations(alphabet)
    tuple_array = tuple(arr)

    permutation_length = len(alphabet)
    found_permutations = set()
    for i in range(len(tuple_array)-permutation_length+1):
        sliding_window = tuple_array[i:i+permutation_length]
        if sliding_window in possible_permutations:
            found_permutations.add(sliding_window)
            if len(found_permutations) == len(possible_permutations):
                return True
    return False

def get_all_permutations(alphabet: Sequence[int])->Sequence[Sequence[int]]:
    # Returns in sorted order if alphabet is sorted
    return list(itertools.permutations(alphabet))

def get_permutation_overlap(permutation1: Sequence[int], permutation2: Sequence[int]) -> int:
    # Counts how many symbols between the two permutations overlap
    if permutation1 == permutation2:
        raise ValueError

    for i in range(len(permutation1)):
        if permutation1[i:]==permutation2[:-i]:
            return len(permutation1)-i
    return 0


def merge_permutations(permutation1: Sequence[int], permutation2: Sequence[int]) -> Sequence[int]:
    overlapping_symbols = get_permutation_overlap(permutation1, permutation2)
    seq = permutation1[:-overlapping_symbols]
    seq = list(seq)
    seq.extend(permutation2)
    return seq
