"""Author: Tove Nordmark
Contains functions for simulating pool shots efficiently"""

import numpy as np
from numba import njit
from . import config
from . import physics


@njit
def action_to_velocity(action):
    """Converts an action of the form ( force/max_force, angle/2pi) into
    a velocity represented as np.array([v_x, v_y])"""
    force = action[0]
    angle = action[1]
    v_x = force * config.MAX_FORCE * np.cos(angle * 2 * np.pi)
    v_y = force * config.MAX_FORCE * np.sin(angle * 2 * np.pi)
    
    return np.array((v_x, v_y))


@njit
def simulate(state, pocketed, action):
    """Runs a simulation of a pool game

    Parameters:
    -----------
    state: numpy.array(shape=(num_balls, 2), dtype=FLOAT_TYPE)
        Axis 0 is the ball index. index = 0 denotes the white ball
        Axis 1 contains the initial ball coordinates (x, y)
    pocketed: numpy.array(shape=(num_balls), dtype=np.bool_)
        An array with values 1 at index i if corresponding ball is pocketed
    action: numpy.array(shape=(2), dtype=FLOAT_TYPE)
        The first value is the force applied to the white ball (between 0 and 1)
        The second value is the angle to the positive x axis as a fraction of 2 pi
    
    Returns:
    --------

    tuple
        numpy.array(shape=(num_balls, 2), dtype=FLOAT_TYPE)
            Axis 0 is the ball index. index = 0 denotes the white ball
            Axis 1 contains the final ball coordinates (x, y)
        numpy.array(shape=(num_balls), np.bool_)
            An array with values 1 at index i if corresponding ball is pocketed
        int
            The total number of collisions during the simulation"""
    n_balls = state.shape[0]
    pocketed = pocketed.copy()
    total_num_collisions = 0
    balls = np.zeros((n_balls, 2, 2), config.FLOAT_TYPE)
    for i in range(n_balls):
        balls[i, 0, :] = state[i, :] # Initial position
    balls[0, 1, :] = action_to_velocity(action) # White ball is moving
    while (balls[:, 1, :] != 0.0).any(): # while balls are still moving
        pocketed = physics.update_pocketed(balls, pocketed)
        balls = physics.move_balls(balls, pocketed)
        balls, num_collisions = physics.apply_collisions(balls, config.LINES, config.LENGTHS, pocketed)
        total_num_collisions += num_collisions
    return balls[:, 0, :], pocketed, total_num_collisions

# NOTE simulation loop not njitted when saving states! 
def simulate_and_save_states(state, pocketed, action):
    """Runs a simulation of a pool game and returns the simulation
    as arrays indexed by time

    Parameters:
    -----------
    state: numpy.array(shape=(num_balls, 2), dtype=FLOAT_TYPE)
        Axis 0 is the ball index. index = 0 denotes the white ball
        Axis 1 contains the initial ball coordinates (x, y)
    pocketed: numpy.array(shape=(num_balls), dtype=np.bool_)
        An array with values 1 at index i if corresponding ball is pocketed
    action: numpy.array(shape=(2), dtype=FLOAT_TYPE)
        The first value is the force applied to the white ball (between 0 and 1)
        The second value is the angle to the positive x axis as a fraction of 2 pi
    
    Returns:
    --------

    tuple
        numpy.array(shape=(n_times, num_balls, 2), dtype=FLOAT_TYPE)
            Axis 0 is the time index
            Axis 1 is the ball index. index = 0 denotes the white ball
            Axis 2 contains the final ball coordinates (x, y)
        numpy.array(shape=(n_times, num_balls), np.bool_)
            Axis 0 is the time index
            Axis 1 is ball index
            (t, i) is True if pocketed at time t, else False
        """
    states = []
    pocketed_over_time = []
    n_balls = state.shape[0]
    pocketed = pocketed.copy()
    total_num_collisions = 0
    balls = np.zeros((n_balls, 2, 2), config.FLOAT_TYPE)
    for i in range(n_balls):
        balls[i, 0, :] = state[i, :] # Initial position
    balls[0, 1, :] = action_to_velocity(action) # White ball is moving
    while (balls[:, 1, :] != 0.0).any(): # while balls are still moving
        states.append(balls[:, 0, :])
        pocketed_over_time.append(pocketed.copy())
        pocketed = physics.update_pocketed(balls, pocketed)
        balls = physics.move_balls(balls, pocketed)
        balls, num_collisions = physics.apply_collisions(balls, config.LINES, config.LENGTHS, pocketed)
        total_num_collisions += num_collisions
    return np.array(states), np.array(pocketed_over_time)







    

