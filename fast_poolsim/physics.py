"""Author: Tove Nordmark
This file contains code for computing collisions, 
updating ball positions and checking if balls are pocketed.
"""
import numpy as np
from numba import njit
from . import config

## Balls to be indexed ball = np.array(shape = (2, 2))
# with ((position), (velocity))

@njit
def dot2(v1, v2):
    """Returns the 2-dimensional dot product of v1 and v2"""
    return v1[0]*v2[0] + v1[1]*v2[1]

@njit
def point_distance(p1, p2):
    """Returns the Euclidean distance between p1 and p2 (2d)"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

@njit
def distance_less_equal(p1, p2, dist):
    """Checks if the Euclidean distance between p1 and p2 is less than dist (True/False)"""
    dist_diff = p1 - p2
    return (dist_diff[0]**2 + dist_diff[1]**2) <= dist**2

@njit
def ball_collision_check(ball1, ball2):
    """Checks if ball1, ball2 are currently colliding (True/False)"""
    pos1 = ball1[0, :]
    pos2 = ball2[0, :]
    vel1 = ball1[1, :]
    vel2 = ball2[1, :]
    # distance check followed by checking if either of the balls are moving
    # followed by vector projection check, to see if both are moving towards
    # each other
    return distance_less_equal(pos1, pos2, 2*config.BALL_RADIUS) and \
    (np.hstack((vel1, vel2)) != 0.0).any() and \
    dot2(pos2 - pos1, vel1 - vel2) > 0

@njit
def collide_balls(ball1, ball2):
    """Return ball1, ball2 (position, velocity) after the current collision
    Should only be run if ball_collision_check is True
    
    Parameters:
    -----------
    ball1 :
        np.array(shape=(2, 2), dtype=config.FLOAT_TYPE)
        Position and velocity of the first ball 
        indices indicate (position/velocity, x/y)
    ball2 :
        np.array(shape = (2, 2), dtype=config.FLOAT_TYPE)
        Position and velocity of the second ball 
        indices indicate (position/velocity, x/y)

    Returns:
    --------
    tuple(np.array(shape=(2, 2), dtype=config.FLOAT_TYPE), np.array(shape=(2, 2), dtype=config.FLOAT_TYPE)) :
        The updated positions and velocities of ball1, ball2 respectively
        after the collision. (indices indicate (position/velocity, x/y))"""
    pos1 = ball1[0, :]
    pos2 = ball2[0, :]
    vel1 = ball1[1, :]
    vel2 = ball2[1, :]
    point_diff = pos2 - pos1
    dist = point_distance(pos1, pos2)
    # normalising circle distance difference vector
    collision = point_diff / dist
    # projecting balls velocity ONTO difference vector
    ball1_dot = dot2(vel1, collision)
    ball2_dot = dot2(vel2, collision)
    # since the masses of the balls are the same, the velocity will just switch
    vel1 += (ball2_dot - ball1_dot) * collision * 0.5*(1+config.BALL_COEFF_OF_RESTITUTION)
    vel2 += (ball1_dot - ball2_dot) * collision * 0.5*(1+config.BALL_COEFF_OF_RESTITUTION)
    # return new balls
    ball1 = np.stack((pos1, vel1), axis=0)
    ball2 = np.stack((pos2, vel2), axis=0)
    return ball1, ball2

@njit
def line_ball_collision_check(line, length, ball):
    """Check if ball is currently colliding with line
    
    Parameters:
    -----------
    line :
        np.array(shape=(2, 2), dtype=config.FLOAT_TYPE)
        line segment from line[0, :] to line[1, :]
    length :
        float
        Length of the line (stored for faster runtime)
    ball :
        np.array(shape=(2, 2), dtype=config.FLOAT_TYPE)
        Position and velocity of the ball 
        indices indicate (position/velocity, x/y)

    Returns:
    --------
    bool
        True if ball is colliding with line, else False"""
    # checks if the ball is half the line length from the line middle
    point1 = line[0, :]
    point2 = line[1, :]
    middle = 0.5 * (point1 + point2)
    
    pos = ball[0, :]
    vel = ball[1, :]
    distance_check = distance_less_equal(middle, pos, length / 2 + config.BALL_RADIUS)
    if distance_check:
        # displacement vector from the first point to the ball
        displacement_to_ball = pos - point1
        # displacement vector from the first point to the second point on the
        # lines
        displacement_to_second_point = point2 - point1
        segment_length = np.sqrt(displacement_to_second_point[0] ** 2 + displacement_to_second_point[1] ** 2)
        normalised_point_diff_vector = displacement_to_second_point / segment_length
        # distance from the first point on the line to the perpendicular
        # projection point from the ball
        projected_distance = dot2(normalised_point_diff_vector, displacement_to_ball)
        # closest point on the line to the ball
        closest_line_point = projected_distance * normalised_point_diff_vector
        perpendicular_vector = np.zeros(2, dtype = config.FLOAT_TYPE)
        perpendicular_vector[0] = -normalised_point_diff_vector[1]
        perpendicular_vector[1] = normalised_point_diff_vector[0]

        # Check conditions
        condition1 = (-config.BALL_RADIUS / 3 <= projected_distance <= length + config.BALL_RADIUS / 3)
        condition2 = point_distance(closest_line_point + point1, pos) <= config.BALL_RADIUS
        condition3 = dot2(perpendicular_vector, vel) <= 0
        return condition1 and condition2 and condition3

        # checking if closest point on the line is actually on the line (which is not always the case when projecting)
        # then checking if the distance from that point to the ball is less than the balls radius and finally
        # checking if the ball is moving towards the line with the dot product
        return (-config.BALL_RADIUS / 3 <= projected_distance and projected_distance <=\
               length + config.BALL_RADIUS / 3) and \
               point_distance(closest_line_point  + line[0, :], pos) <= \
               config.BALL_RADIUS and dot2(
            perpendicular_vector, vel) <= 0
    else:
        return False

@njit
def collide_line_ball(line, length, ball):
    """Return new position and velocity of ball after collision with line
    should only be run if line_ball_collision_check() is True
    
    Parameters:
    -----------
    line :
        np.array(shape=(2, 2), dtype=config.FLOAT_TYPE)
        line segment from line[0, :] to line[1, :]
    length :
        float
        Length of the line (stored for faster runtime)
    ball :
        np.array(shape=(2, 2), dtype=config.FLOAT_TYPE)
        Position and velocity of the ball 
        indices indicate (position/velocity, x/y)

    Returns:
    --------
    np.array(shape=(2, 2), dtype=config.FLOAT_TYPE)
        New position and velocity of the ball after collision
        indices indicate (position/velocity, x/y)"""
    pos = ball[0, :]
    vel = ball[1, :]
    displacement_to_second_point = line[1, :] - line[0, :]
    normalised_point_diff_vector = displacement_to_second_point / \
                                   length
    perpendicular_vector = np.zeros(2, dtype=config.FLOAT_TYPE)
    perpendicular_vector[0] = -normalised_point_diff_vector[1]
    perpendicular_vector[1] = normalised_point_diff_vector[0]

    vel -= 2 * dot2(perpendicular_vector,vel) * \
                     perpendicular_vector * 0.5*(1+config.TABLE_COEFF_OF_RESTITUTION)
    return np.stack((pos, vel), axis=0)

@njit
def update_ball(ball):
    """Return updated ball position and velocity after one time step"""
    pos = ball[0, :]
    vel = ball[1, :]

    vel *= config.FRICTION_COEFF
    pos += vel

    if dot2(vel, vel) < config.SQUARE_FRICTION_THRESHOLD:
        vel = np.zeros(2, dtype = config.FLOAT_TYPE)
    return np.stack((pos, vel), axis = 0)

@njit
def get_active_balls(balls, pocketed):
    """Returns an array containing the non pocketed balls
    and the indices of these balls in the original array
    
    Parameters:
    -----------
    balls : 
        np.array(shape=(n_balls, 2, 2),dtype=config.FLOAT_TYPE)
        The positions and velocities of all balls
        Indices indicate (ball number, position/velocity, x/y)
    pocketed :
        np.array(shape=(n_balls), dtype=np.bool_)
        pocketed[i] == True if ball i is pocketed, else False
    
    Returns:
    --------
    tuple(np.array(shape=(n_active, 2, 2), dtype=config.FLOAT_TYPE), np.array(shape=n_active, dtype=config.INT_TYPE)) :
        The positions and velocities of the active balls and their indices in the
        original array passed to this function.
    """
    pocketed = pocketed.astype(np.bool_)
    n_active = (~pocketed).sum()
    active_balls = np.zeros((n_active, 2, 2), config.FLOAT_TYPE)
    active_ball_indices = np.zeros(n_active, config.INT_TYPE)
    active_ball_index = 0
    for i in range(balls.shape[0]):
        if not pocketed[i]:
            active_ball_indices[active_ball_index] = i
            active_balls[active_ball_index, :, :] = balls[i, :, :]
            active_ball_index += 1
    return active_balls, active_ball_indices

@njit
def apply_collisions(balls, lines, lengths, pocketed):
    """ Check and apply collisions between balls and each other/environment

    Parameters:
    -----------
    balls :
        np.array(shape=(n_balls, 2, 2, dtype=config.FLOAT_TYPE))
        The positions and velocities of all balls
        Indices indicate (ball number, position/velocity, x/y)
    lines :
        np.array(shape=(n_lines, 2, 2), dtype=config.FLOAT_TYPE)
        line segment i goes from line[i, 0, :] to line[i, 1, :]
    lengths :
        np.array(shape=(n_lines), dtype=config.FLOAT_TYPE)
        Lengths of the lines (stored for faster runtime)
    pocketed :
        np.array(shape=(n_balls), dtype=np.bool_)
        pocketed[i] == True if ball i is pocketed, else False
    
    Returns:
    --------
    tuple(np.array(shape=(n_balls, 2, 2, dtype=config.FLOAT_TYPE)), int)
        The updateds positions and velocities of the balls 
        and the number of collisions occurring, respectively
    """
    balls = balls.copy() # scared of modifying original array
    pocketed = pocketed.astype(np.bool_)
    active_balls, active_ball_indices = get_active_balls(balls, pocketed)
    n_lines = lines.shape[0]
    n_active = active_balls.shape[0]
    num_collisions = 0
    # Go through combinations of balls
    for i in range(n_active):
        for j in range(i):
            ball_i = active_balls[i, :, :]
            ball_j = active_balls[j, :, :]
            if ball_collision_check(ball_i, ball_j):
                # Collide balls
                new_ball_i, new_ball_j = collide_balls(ball_i, ball_j)
                balls[active_ball_indices[i], :, :] = new_ball_i
                balls[active_ball_indices[j], :, :] = new_ball_j
                num_collisions += 1
    # Go through ball, line combinations
    for i in range(n_active):
        ball = active_balls[i, :, :]
        for j in range(n_lines):
            line = lines[j, :, :]
            length = lengths[j]
            if line_ball_collision_check(line, length, ball): 
                # Collide balls, lines
                balls[active_ball_indices[i], :, :] = collide_line_ball(line, length, ball)
                num_collisions += 1
                break # Dont't check remaining lines
    return balls, num_collisions

@njit
def move_balls(balls, pocketed):
    """ Move balls one time step

    Parameters:
    -----------
    balls :
        np.array(shape=(n_balls, 2, 2, dtype=config.FLOAT_TYPE))
        The positions and velocities of all balls
        Indices indicate (ball number, position/velocity, x/y)
    pocketed :
        np.array(shape=(n_balls), dtype=np.bool_)
        pocketed[i] == True if ball i is pocketed, else False
    
    Returns:
    --------
    np.array(shape=(n_balls, 2, 2, dtype=config.FLOAT_TYPE))
        The updated positions and velocities of the balls 
    """
    balls = balls.copy() # scared of modifying original array
    pocketed = pocketed.astype(np.bool_)
    for i in range(balls.shape[0]): # Go through balls
        # Update positions
        if not pocketed[i]: # Active
            balls[i, :, :] = update_ball(balls[i, :, :])
        else: # Pocketed
            balls[i, 1, :] = 0.0 
    return balls


@njit
def update_pocketed(balls, pocketed):
    """Checks if any new balls are pocketed and updates the pocketed
    array accordingly

    Parameters:
    -----------
    balls :
        np.array(shape=(n_balls, 2, 2, dtype=config.FLOAT_TYPE))
        The positions and velocities of all balls
        Indices indicate (ball number, position/velocity, x/y)
    pocketed :
        np.array(shape=(n_balls), dtype=np.bool_)
        pocketed[i] == True if ball i is pocketed, else False
    
    Returns:
    --------
    np.array(shape=(n_balls, 2, 2, dtype=np.bool_))
        The updated array of pocketed balls
    """
    pocketed = pocketed.astype(np.bool_)
    active_balls, active_ball_indices = get_active_balls(balls, pocketed)
    for i in range(active_balls.shape[0]):
        for j in range(config.HOLES.shape[0]):
            if distance_less_equal(active_balls[i, 0, :], config.HOLES[j, :], config.HOLE_RADIUS):
                pocketed[active_ball_indices[i]] = True
                break # ball cannot be in multiple holes at once
    return pocketed