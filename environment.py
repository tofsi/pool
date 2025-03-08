from typing import Optional
import numpy as np
import gymnasium as gym

class PoolEnvironment(gym.Env):

    def __init__(self):
        print("Initialised")

    def step(action): # An action is a tuple of (angle, share_of_max_force). 
        (angle, force) = action

        # some code to take action

        reward = 0 # compute reward properly
        

