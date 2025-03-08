from typing import Optional
import numpy as np
import gymnasium as gym

from play_pool import simulate

# Our observation space is [array of positions] + [pocketed balls]
class PoolEnvironment(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, ball_count = 2):
                # Instance attributes
        self.ball_count = ball_count
        self.balls = []  # Array of (x,y) positions
        self.pocketed = np.zeros(ball_count)  # Categorical indicators for balls pocketed
        self.initial_ball_positions = []
        
        # Define action space - angle and force
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([1, 2*np.pi]), 
            shape = (2,),
            dtype=np.float32
        )
        
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(3 * ball_count,), 
            dtype=np.float32
        )   
        
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
        all_non_white_balls_pocketed = np.sum(pocketed) == self.ball_count - 1 and not white_ball_pocketed


        if white_ball_pocketed:
            reward = -1 
            terminated = True
            observation = (self.balls, self.pocketed)
        else:
            reward = np.sum(newly_pocketed) / collisions
            terminated = all_non_white_balls_pocketed
            observation = (self.balls, self.pocketed)
        
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.balls = self.initial_ball_positions.copy()
        self.pocketed = np.zeros(self.ball_count)


