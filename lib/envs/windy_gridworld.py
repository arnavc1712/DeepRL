import gym
import numpy as np
import sys
from gym import spaces

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridworldEnv(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7)
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self):
        self.shape = (7, 10)

        self.nS = np.prod(self.shape)
        self.nA = 4

        self.observation_space = spaces.Box(low = np.array([0]),
                                            high = np.array([self.nS-1]),
                                            dtype = np.int32)

        self.action_space = spaces.Discrete(self.nA,)

        # Wind strength
        winds = np.zeros(self.shape)
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        # Calculate transition probabilities
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = { a : [] for a in range(self.nA) }
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        self.S = np.ravel_multi_index((3,0), self.shape)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        _, self.S, reward, is_done = self.P[self.S][action][0]
        return self.S, reward, is_done, {}
    
    def reset(self):
        self.S = np.ravel_multi_index((3,0), self.shape)
        return self.S

    def render(self, mode='human'):
        self._render(mode)

    def _render(self, mode='human'):

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.S == s:
                output = " x "
            elif position == (3,7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")