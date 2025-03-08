from typing import Optional
import numpy as np
import gymnasium as gym

from play_pool import simulate, get_initial_state

# Our observation space is [array of positions] + [pocketed balls]
class PoolEnvironment(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def get_flat_obs(self):
        flat_obs = np.zeros(self.ball_count * 3, dtype=np.float32)
        for i in range(self.ball_count):
            base_idx = i * 3
            if i < len(self.balls):
                flat_obs[base_idx] = self.balls[i][0]     
                flat_obs[base_idx + 1] = self.balls[i][1] 
        
            flat_obs[base_idx + 2] = self.pocketed[i]
    
        return flat_obs


    def __init__(self, ball_count = 2):
                # Instance attributes
        self.ball_count = ball_count
        self.balls = get_initial_state(ball_count) # Array of (x,y) positions
        self.pocketed = np.zeros(ball_count)  # Categorical indicators for balls pocketed
        
        # Define action space - angle and force
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([1, 2*np.pi]), 
            shape = (2,),
            dtype=np.float32
        )
        
        
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, 0] * ball_count),
            high=np.array([np.inf, np.inf, 1] * ball_count),
            shape=(ball_count * 3,),
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
        self.balls = new_ball_positions
        white_ball_pocketed = newly_pocketed[0] == 1 # This is a terminal state
        all_non_white_balls_pocketed = np.sum(pocketed) == self.ball_count - 1 and not white_ball_pocketed

        if white_ball_pocketed:
            reward = -1 
            terminated = True
        else:
            reward = np.sum(newly_pocketed) / collisions
            terminated = all_non_white_balls_pocketed
        
        return self.get_flat_obs(), reward, terminated, truncated, info

    def reset(self):
        self.balls = get_initial_state(self.ball_count)
        self.pocketed = np.zeros(self.ball_count)
        info = {}
    
        return self.get_flat_obs(), info
    
    