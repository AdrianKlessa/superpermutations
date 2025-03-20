import numpy as np
from permutation_env import PermutationEnvironment
import permutation_utils
import gymnasium as gym
from gymnasium import spaces


class GymPermutationEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, alphabet_size=4, render_mode="console", truncate_episodes=True):
        super(GymPermutationEnv, self).__init__()
        self.render_mode = render_mode  # Basically console-only for this use case
        self.truncate_episodes = truncate_episodes

        # Set up custom variables for tracking permutations
        self.alphabet = list(range(1, alphabet_size + 1))
        env = PermutationEnvironment(self.alphabet)
        self.env = env
        self.step_count = 0
        self.alphabet_size = alphabet_size
        # Action space
        n_actions = env.number_of_possible_permutations
        self.action_space = spaces.Discrete(n_actions)

        # Observation space
        # n = alphabet_size!
        # Binary 1d array of length n indicating which permutations were added already
        # Concatenated with same-sized one-hot encoding of which permutation was added last
        self.observation_space = spaces.MultiDiscrete([2] * (n_actions * 2))

        self.bound_sizes = {
        2: 3,
        3: 9,
        4: 33,
        5: 153,
        6: 872,
        7: 5906,
        8: 46205,
        9: 408966
    }

    def reset(self, seed=None, options=None):
        self.env = PermutationEnvironment(self.alphabet)
        self.step_count = 0
        super().reset(seed=seed, options=options)
        return self.env.get_observation(), {}

    def step(self, action):
        if self.env.permutations_added_array[action]:
            reward = -1
        else:
            existing_permutation = self.env.state
            if existing_permutation:
                permutations_before = self.env.permutations_added_array.copy()
                added_permutation = self.env.possible_permutations[action]
                reward = permutation_utils.get_permutation_overlap(existing_permutation, list(added_permutation))
                self.env.add_permutation(action)

                # Deal with the case when merging permutation A&B creates a string with A,B and C
                permutations_after = self.env.permutations_added_array
                permutations_diff = np.bitwise_xor(permutations_after, permutations_before)
                number_of_differences = np.count_nonzero(permutations_diff)
                if number_of_differences > 1:
                    reward += len(self.env.alphabet)*(number_of_differences-1)
            else:
                # No overlap since this is the first permutation to be added
                reward = 0
                self.env.add_permutation(action)
            reward/=self.alphabet_size # Normalize reward

        terminated = permutation_utils.is_superpermutation(np.array(self.env.state),
                                                           self.alphabet)  # TODO: Remove casting
        self.step_count += 1
        truncated = False
        # Step count is number of added permutations (minus useless actions by the agent)
        # Bound_size[alphabet] is in characters so for an optimal superpermutation step_count will be much smaller
        if self.truncate_episodes and self.step_count > self.bound_sizes[self.alphabet_size]:
            truncated = True

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
