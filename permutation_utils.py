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


def get_permutation_ids_contained_in_symbols_string(symbols_string: Sequence[int], alphabet: Sequence[int]) -> Sequence[
    int]:
    """
    Get a list of permutations found in a string of symbols
    :param symbols_string: String of symbols from a given alphabet
    :param alphabet: List of possible symbols
    :return: ids of the permutations found in the symbols string
    """
    possible_permutations = get_all_permutations(alphabet)
    tuple_array = tuple(symbols_string)

    permutation_length = len(alphabet)
    found_permutations = set()
    found_ids = set()
    for i in range(len(tuple_array) - permutation_length + 1):
        sliding_window = tuple_array[i:i + permutation_length]
        if sliding_window in possible_permutations:
            found_permutations.add(sliding_window)
            found_ids.add(possible_permutations.index(sliding_window))
            if len(found_permutations) == len(possible_permutations):
                return list(found_ids)  # we already found all possible permutations
    return list(found_ids)

def get_possible_relabellings(symbols_string: Sequence[int], alphabet: Sequence[int]) -> Sequence[Sequence[int]]:
    possible_permutations = get_all_permutations(alphabet)
    labellings = []
    for permutation in possible_permutations:
        seq = []
        for symbol in symbols_string:
            seq.append(permutation[symbol-1]) # 0 not used as a symbol so -1
        labellings.append(seq)
    return labellings


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
    if overlapping_symbols == 0:
        seq = list(permutation1)
    else:
        seq = list(permutation1[:-overlapping_symbols])
    seq.extend(permutation2)
    return seq


def check_inform_length(alphabet_size, superpermutation):
    superpermutation_length = len(superpermutation)
    superpermutation_sizes = {
        2: 3,
        3: 9,
        4: 33,
        5: 153,
        6: 872,
        7: 5906,
        8: 46205,
        9: 408966
    }

    if superpermutation_length <= superpermutation_sizes[alphabet_size]:
        print(f"Below/at upper bound found for n={alphabet_size}:")
        print(superpermutation)
    elif superpermutation_length <= (superpermutation_sizes[alphabet_size] + 10):
        print(f"Close to upper bound (but above) for n={alphabet_size}:")
        print(superpermutation)
