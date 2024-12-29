from typing import Sequence

import numpy as np
import numpy.typing as npt
import itertools


def is_superpermutation(arr: npt.NDArray[np.int_], alphabet: Sequence[int]) -> bool:
    """
    Check if a given numpy array is a superpermutation of the given alphabet.
    :param arr:
    :param alphabet:
    :return:
    """
    possible_permutations = get_all_permutations(alphabet)
    tuple_array = tuple(arr)

    permutation_length = len(alphabet)
    found_permutations = set()
    for i in range(len(tuple_array) - permutation_length + 1):
        sliding_window = tuple_array[i:i + permutation_length]
        if sliding_window in possible_permutations:
            found_permutations.add(sliding_window)
            if len(found_permutations) == len(possible_permutations):
                return True
    return False


def get_all_permutations(alphabet: Sequence[int]) -> Sequence[Sequence[int]]:
    """
    Returns permutations in sorted order if the alphabet is sorted
    :param alphabet:
    :return:
    """
    return list(itertools.permutations(alphabet))


def get_permutation_overlap(permutation1: Sequence[int], permutation2: Sequence[int]) -> int:
    """
    Counts how many symbols between the two permutations overlap
    :param permutation1:
    :param permutation2:
    :return:
    """
    if permutation1 == permutation2:
        raise ValueError
    if len(permutation1) > len(permutation2):
        permutation1 = permutation1[-len(permutation2):]

    for i in range(len(permutation1)):
        if permutation1[i:] == permutation2[:-i]:
            return len(permutation1) - i
    return 0


def merge_permutations(permutation1: Sequence[int], permutation2: Sequence[int]) -> Sequence[int]:
    """
    Merge permutations, e.g. [1,2],[2,1] --> [1,2,1]
    Can also merge the the tail of a part of a superpermutation with a new permutation
    e.g. [2,3,1,2,3],[3,2,1] --> [2,3,1,2,3,2,1]
    :param permutation1:
    :param permutation2:
    :return:
    """
    overlapping_symbols = get_permutation_overlap(permutation1, permutation2)
    if overlapping_symbols==0:
        seq = list(permutation1)
    else:
        seq = list(permutation1[:-overlapping_symbols])
    seq.extend(permutation2)
    return seq
