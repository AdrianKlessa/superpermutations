### Superpermutations

This repository contains various scripts related to looking for the shortest superpermutations, aka the *Haruhi problem*

The `superpermutations for n4.ipynb` file contains a proof-of-concept of using the Proximal Policy Optimization RL algorithm to find the shortest superpermutation on an alphabet with 4 symbols.

### Tools used:

* Gymnasium API was used to create a reinforcement learning environment compatible with other libraries
* Stable Baselines3 was used to train RL agents on the aforementioned environment

### Included files:

* `permutation_utils` contains functions to get all possible permutations of an alphabet, checking how much two permutations overlap, checking if a string is a valid superpermutation etc.
* `permutation_env` contains a custom class that contains all necessary information and methods for building a superpermutation step-by-step by appending (with overlaps) new permutations
* `GymPermutationEnv` is a wrapper over `permutation_env` providing support for the Gymnasium API used by Stable Baselines3
* `superpermutations for n4` is a Jupyter notebook with a POC for generating the shortest superpermutation for n=4

### Observation / action format

Given an alphabet with `n` distinct characters each of the `n!` permutations is assigned an ID. 

Then, an observation is composed of two concatenated 1D arrays of length `n!` - the first has 1s only at the indices corresponding to previously added (either through direct addition or by an overlap between a previous and new permutation) permutations, the second has a single `1` entry at the index of the **last** added permutation.