import numpy as np
from permutation_env import PermutationEnvironment
import permutation_utils
import gymnasium as gym
from gymnasium import spaces


class GymPermutationEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, alphabet_size=4, render_mode="console"):
        super(GymPermutationEnv, self).__init__()
        self.render_mode = render_mode  # Basically console-only for this use case

        # Set up custom variables for tracking permutations
        self.alphabet = list(range(1, alphabet_size + 1))
        env = PermutationEnvironment(self.alphabet)
        self.permutation_env = env

        # Action space
        n_actions = env.number_of_possible_permutations
        self.action_space = spaces.Discrete(n_actions)

        # Observation space
        # n = alphabet_size!
        # Binary 1d array of length n indicating which permutations were added already
        # Concatenated with same-sized one-hot encoding of which permutation was added last
        self.observation_space = spaces.MultiDiscrete([2] * (n_actions * 2))

    def reset(self, seed=None, options=None):
        self.env = PermutationEnvironment(self.alphabet)
        super().reset(seed=seed, options=options)
        return self.env.get_observation(), {}

    def step(self, action):
        # TODO: Check if adding a permutation doesn't add a third permutation as well
        # TODO: e.g. [1,2,3,4,5]+[3,4,5,1,2]-->[1,2,3,4,5,1,2]
        # TODO: which contains both [1,2,3,4,5],[3,4,5,1,2] as well as [2,3,4,5,1]
        # TODO: If yes, then add reward equal to the permutation length
        if self.env.permutations_added_array[action]:
            reward = -5
        else:
            existing_permutation = self.env.state
            if existing_permutation:
                added_permutation = self.env.possible_permutations[action]
                reward = permutation_utils.get_permutation_overlap(existing_permutation, list(added_permutation))
                self.env.add_permutation(action)
            else:
                # No overlap since this is the first permutation to be added
                reward = 0
                self.env.add_permutation(action)

        terminated = permutation_utils.is_superpermutation(np.array(self.env.state),
                                                           self.alphabet)  # TODO: Remove casting
        truncated = False
        # TODO: Give reward for finishing superpermutation?
        if terminated:
            permutation_utils.check_inform_length(len(self.alphabet), self.env.state)

        info = {}

        return (
            self.env.get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.render_mode == "console":
            print(self.env.state)

    def close(self):
        pass
