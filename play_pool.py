from poolsim.pool import ball
from poolsim.pool import collisions
from poolsim.pool import config
from poolsim.pool import cue
from poolsim.pool import event
from poolsim.pool import gamestate
from poolsim.pool import graphics
from poolsim.pool import physics
from poolsim.pool import table_sprites


import numpy as np
import pandas as pd
from itertools import combinations
_nballs = 5
_max_force = 1


def action_to_velocity(action):
    force = action[0]
    angle = action[1]
    x = force * _max_force * np.cos(angle * 2 * np.pi)
    y = force * _max_force * np.sin(angle * 2 * np.pi)
    return np.array((x, y))

def set_up_table():
    # TODO: generate list of table_sprites.TableSide() instances for checking collisions etc.
    return None

def simulate(state, pocketed, action):
    # Should return next_state, next_pocketed, n_collisions
    num_collisions = 0
    balls = np.array([ball.Ball() for k in range(_nballs + 1)]) # Initialize board state
    balls[0].set_new_velocity(action_to_velocity(action)) # Velocity of white ball
    table_sides = np.array([table_sprites.TableSide()])
    all_stationary = False
    while not all_stationary:
        # Move balls
        for ball in balls:
            ball.update()
            # Apply collisions between ball, wall
            if physics.line_ball_collision_check():
                pass

        # Apply collisions between balls
        for ball, other_ball in combinations(balls, 2):
            if physics.ball_collision_check(ball, other_ball):
                physics.collide_balls(ball, other_ball)
                num_collisions += 1 # Add for all collisions


        
        
    
    

if __name__=="__main__":
    state = np.array((1 + _nballs, 2), dtype = np.float32)
    simulate(state)





