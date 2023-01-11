import numpy as np
import sys
from gym import spaces
import gym


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class CliffWalkingEnv(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        reward = -100.0 if self._cliff[tuple(new_position)] else -1.0
        is_done = self._cliff[tuple(new_position)] or (tuple(new_position) == (3,11))
        return [(1.0, new_state, reward, is_done)]

    def __init__(self):
        self.shape = (4, 12)

        self.nS = np.prod(self.shape)
        self.nA = 4

        self.observation_space = spaces.Box(low = np.array([0]),
                                            high = np.array([self.nS-1]),
                                            dtype = np.int32)

        self.action_space = spaces.Discrete(self.nA,)


        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = { a : [] for a in range(self.nA) }
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

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
            elif position == (3,11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip() 
            if position[1] == self.shape[1] - 1:
                output = output.rstrip() 
                output += "\n"

            outfile.write(output)
        outfile.write("\n")