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

if __name__=="__main__":
    state = np.array([[500,
                        200], [550,200],
                        [550,205]], dtype=config.FLOAT_TYPE)
    pocketed = np.zeros(3, np.bool_)
    action = np.array([1.0, 0], config.FLOAT_TYPE)
    print(simulate(state, pocketed, action))






    

