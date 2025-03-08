from typing import Optional
import numpy as np
import gymnasium as gym

from play_pool import simulate

# Our observation space is [array of positions] + [pocketed balls]
class PoolEnvironment(gym.Env):

    ball_positions = [] # Array of (x,y) positions
    pocketed = [] # Categorical indicators for if balls are pocketed
    ball_count = 0 # Count of balls, includes white ball
    
    initial_ball_positions = []
    
    action_space = gym.spaces.Box(low = [0,0], high = [1, 2*np.pi],shape=(2,), dtype = np.float32)
    # Observation space is 
    position_space = gym.spaces.Box(low = [0,0], high = [2,1], shape = (ball_count,2), dtype = np.float32)
    pocketed_space = gym.spaces.Box(low = 0, high = 1, shape =(ball_count,),dtype=np.int32)
    observation_space = gym.spaces.Dict(positions = position_space, pocketed = pocketed_space)


    def __init__(self):
        print("Initialised")

    def step(self, action): # An action is a tuple of (angle, share_of_max_force). 
        (angle, force) = action

        observation = ()
        reward = 0
        terminated = False
        truncated = False
        info = ""

        new_ball_positions, newly_pocketed, collisions = simulate(self.balls,self.pocketed, action)

        pocketed = pocketed + newly_pocketed
        self.ball_positions = new_ball_positions
        white_ball_pocketed = newly_pocketed[0] == 1 # This is a terminal state
        all_non_white_balls_pocketed = sum(pocketed) == self.ball_count - 1 and not white_ball_pocketed


        if white_ball_pocketed:
            reward = -1 
            terminated = True
            observation = (self.balls, self.pocketed)
        else:
            reward = sum(newly_pocketed) / collisions
            terminated = all_non_white_balls_pocketed
            observation = (self.balls, self.pocketed)
        
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.ball_positions = self.initial_ball_positions.copy()
        self.pocketed = np.zeros(self.ball_count)


