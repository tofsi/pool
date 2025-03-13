import numpy as np
from numba import njit
from . import config

## Balls to be indexed ball = np.array(shape = (2, 2))
# with ((position), (velocity))

@njit
def dot2(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

@njit
def point_distance(p1, p2):
    """Euclidean distance between p1 and p2 (2d)"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

@njit
def distance_less_equal(p1, p2, dist):
    dist_diff = p1 - p2
    return (dist_diff[0]**2 + dist_diff[1]**2) <= dist**2

@njit
def ball_collision_check(ball1, ball2):
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
    """Return ball1, ball2 (position, velocity) after collision"""
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
def rotation_matrix(axis, theta):
    # Return the rotation matrix associated with counterclockwise rotation about
    # the given axis by theta radians.
    axis = axis / np.sqrt(dot2(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

@njit
def line_ball_collision_check(line, length, ball):
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
    """Return new ball (position, velocity) after collision"""
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
    """Return updated ball positions after one time step"""
    pos = ball[0, :]
    vel = ball[1, :]

    vel *= config.FRICTION_COEFF
    pos += vel

    if dot2(vel, vel) < config.SQUARE_FRICTION_THRESHOLD:
        vel = np.zeros(2, dtype = config.FLOAT_TYPE)
    return np.stack((pos, vel), axis = 0)

@njit
def get_active_balls(balls, pocketed):
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
    pocketed = pocketed.astype(np.bool_)
    active_balls, active_ball_indices = get_active_balls(balls, pocketed)
    for i in range(active_balls.shape[0]):
        for j in range(config.HOLES.shape[0]):
            if distance_less_equal(active_balls[i, 0, :], config.HOLES[j, :], config.HOLE_RADIUS):
                pocketed[active_ball_indices[i]] = True
                break # ball cannot be in multiple holes at once
    return pocketed