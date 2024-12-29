### Superpermutations

This repository contains various scripts related to looking for the shortest superpermutations, aka the *Haruhi problem*

### Tools used:

* Gymnasium API was used to create a reinforcement learning environment compatible with other libraries
* Stable Baselines3 was used to train RL agents on the aforementioned environment

#### TODO:

* Modify the RL environment to detect when merging two permutations yields a string containing both the original permutations, as well as a third new permutation.
  *  e.g. [1,2,3,4,5]+[3,4,5,1,2]-->[1,2,3,4,5,1,2]