import gym
from gym import spaces
import numpy as np
import io
import sys

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(gym.Env):

    def __init__(self):
        self.shape = (4,4)

        self.nS = np.prod(self.shape)
        self.nA = 4

        self.observation_space = spaces.Box(low = np.array([0]),
                                            high = np.array([self.nS-1]),
                                            dtype = np.int32)

        self.action_space = spaces.Discrete(self.nA,)

        MAX_Y = self.shape[0]
        MAX_X = self.shape[1]

        P = {}
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(self.nA)}

            is_done = lambda s: s == 0 or s == (self.nS - 1)
            reward = 0.0 if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(self.nS) / self.nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        self.reset()
    

    def get_action_meanings(self):
        return {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}
    
    def reset(self):
        self.S = 8 # (3,0)
        return self.S

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        prob, self.S, reward, is_done = self.P[self.S][action][0]
        return self.S, reward, is_done, {}

    def render(self, mode='human'):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.S == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()