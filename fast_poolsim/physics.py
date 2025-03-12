import numpy as np
from numba import njit
from fast_poolsim import config

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
    return distance_less_equal(ball1, ball2, 2*config._ball_radius) and \
    (np.hstack((vel1, vel2)) != 0.0).any() and \
    dot2(pos2 - pos1, vel1 - vel2) > 0

@njit
def collide_balls(ball1, ball2):
    pos1 = ball1[0, :]
    pos2 = ball2[0, :]
    vel1 = ball1[1, :]
    vel2 = ball2[1, :]
    point_diff = pos2 - pos1
    dist = point_distance(pos1, pos2)
    # normalising circle distance difference vector
    collision = point_diff / dist
    # projecting balls velocity ONTO difference vector
    ball1_dot = dot2(ball1.velocity, collision)
    ball2_dot = dot2(ball2.velocity, collision)
    # since the masses of the balls are the same, the velocity will just switch
    vel1 += (ball2_dot - ball1_dot) * collision * 0.5*(1+config.ball_coeff_of_restitution)
    vel2 += (ball1_dot - ball2_dot) * collision * 0.5*(1+config.ball_coeff_of_restitution)
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
    distance_check = distance_less_equal(middle, pos, length / 2 + config.ball_radius)
    if distance_check:
        # displacement vector from the first point to the ball
        displacement_to_ball = pos - point1
        # displacement vector from the first point to the second point on the
        # lines
        displacement_to_second_point = point2 - point1
        
        normalised_point_diff_vector = displacement_to_second_point / length
        # distance from the first point on the line to the perpendicular
        # projection point from the ball
        projected_distance = dot2(normalised_point_diff_vector, displacement_to_ball)
        # closest point on the line to the ball
        closest_line_point = projected_distance * normalised_point_diff_vector
        perpendicular_vector = np.zeros(2, dtype = np.float64)
        perpendicular_vector[0] =-normalised_point_diff_vector[1]
        perpendicular_vector[1] =normalised_point_diff_vector[0]

        # checking if closest point on the line is actually on the line (which is not always the case when projecting)
        # then checking if the distance from that point to the ball is less than the balls radius and finally
        # checking if the ball is moving towards the line with the dot product
        return -config.ball_radius / 3 <= projected_distance <= \
               length + config.ball_radius / 3 and \
               np.hypot(*(closest_line_point - pos + line[0])) <= \
               config.ball_radius and np.dot(
            perpendicular_vector, vel) <= 0