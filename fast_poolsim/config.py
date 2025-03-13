"""Author: Tove Nordmark
This file defines the simulation environment through various constants."""
import numpy as np
from itertools import product

FLOAT_TYPE = np.float32
INT_TYPE = np.int16

MAX_FORCE = 15

TABLE_MARGIN = 40
RESOLUTION = (1000, 500)
BALL_RADIUS = 14
# physics
# if the velocity of the ball is less then
# friction threshold then it is stopped
FRICTION_THRESHOLD = 0.06
SQUARE_FRICTION_THRESHOLD = FRICTION_THRESHOLD ** 2
FRICTION_COEFF = 0.99
# 1 - perfectly elastic ball collisions
# 0 - perfectly inelastic collisions
BALL_COEFF_OF_RESTITUTION = 0.9
TABLE_COEFF_OF_RESTITUTION = 0.9

HOLE_RADIUS =  22
MIDDLE_HOLE_OFFSET = np.array([[-HOLE_RADIUS * 2, HOLE_RADIUS], [-HOLE_RADIUS, 0],
                               [HOLE_RADIUS, 0], [HOLE_RADIUS * 2, HOLE_RADIUS]]).astype(INT_TYPE)
SIDE_HOLE_OFFSET = np.array([
    [- 2 * np.cos(np.radians(45)) * HOLE_RADIUS - HOLE_RADIUS, HOLE_RADIUS],
    [- np.cos(np.radians(45)) * HOLE_RADIUS, -
    np.cos(np.radians(45)) * HOLE_RADIUS],
    [np.cos(np.radians(45)) * HOLE_RADIUS,
     np.cos(np.radians(45)) * HOLE_RADIUS],
    [- HOLE_RADIUS, 2 * np.cos(np.radians(45)) * HOLE_RADIUS + HOLE_RADIUS]
]).astype(FLOAT_TYPE)

def setup_table():
    """ Sets up the pool table excluding balls

    Returns:
    --------
    tuple
        numpy.array(shape=(n_holes, 2), dtype=FLOAT_TYPE)
            An array containing the positions of holes on the board
            axis 0: hole index
            axis 1: coordinates
        numpy.array(shape=(n_lines, 2, 2), dtype=FLOAT_TYPE)
            An array containing the lines delimiting the table
            axis 0: line number
            axis 1: start/end point
            axis 2: x/y coordinates
        numpy.array(shape=(n_holes), dtype=FLOAT_TYPE)
            An array containing the lengths of the lines delimiting
            the table (stored because lengths are expensive to recompute)
    """
    lines = []
    lengths = []
    table_side_points = np.empty((1, 2))
    # holes_x and holes_y holds the possible xs and ys of the table holes
    # with a position ID in the second tuple field
    # so the top left hole has id 1,1
    holes_x = [(TABLE_MARGIN, 1), (RESOLUTION[0] /
                                            2, 2), (RESOLUTION[0] - TABLE_MARGIN, 3)]
    holes_y = [(TABLE_MARGIN, 1),
                (RESOLUTION[1] - TABLE_MARGIN, 2)]
    # next three lines are a hack to make and arrange the hole coordinates
    # in the correct sequence
    all_hole_positions = np.array(
        list(product(holes_y, holes_x)))
    all_hole_positions = np.fliplr(all_hole_positions)
    all_hole_positions = np.vstack(
        (all_hole_positions[:3], np.flipud(all_hole_positions[3:])))
    for hole_pos in all_hole_positions:
        # this will generate the diagonal, vertical and horizontal table
        # pieces which will reflect the ball when it hits the table sides
        #
        # they are generated using 4x2 offset matrices (4 2d points around the hole)
        # with the first point in the matrix is the starting point and the
        # last point is the ending point, these 4x2 matrices are
        # concatenated together
        #
        # the martices must be flipped using numpy.flipud()
        # after reflecting them using 2x1 reflection matrices, otherwise
        # starting and ending points would be reversed
        if hole_pos[0][1] == 2:
            # hole_pos[0,1]=2 means x coordinate ID is 2 which means this
            # hole is in the middle
            offset = MIDDLE_HOLE_OFFSET
        else:
            offset = SIDE_HOLE_OFFSET
        if hole_pos[1][1] == 2:
            offset = np.flipud(offset) * [1, -1]
        if hole_pos[0][1] == 1:
            offset = np.flipud(offset) * [-1, 1]
        table_side_points = np.append(
            table_side_points, [hole_pos[0][0], hole_pos[1][0]] + offset, axis=0)
    # deletes the 1st point in array (leftover form np.empty)
    table_side_points = np.delete(table_side_points, 0, 0)
    for num, point in enumerate(table_side_points[:-1]):
        # this will skip lines inside the circle
        if num % 4 != 1:
            lines.append(
                [point, table_side_points[num + 1]])
            lengths.append(np.sqrt(np.dot(point - table_side_points[num + 1], point - table_side_points[num + 1])))
    lines.append((
        [table_side_points[-1], table_side_points[0]]))
    lengths.append(np.sqrt(np.dot(table_side_points[-1] - table_side_points[0], table_side_points[-1] - table_side_points[0])))
    return  all_hole_positions[:, :, 0].astype(FLOAT_TYPE), np.array(lines).astype(FLOAT_TYPE), np.array(lengths).astype(FLOAT_TYPE)

HOLES, LINES, LENGTHS = setup_table()