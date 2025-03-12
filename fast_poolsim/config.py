import numpy as np

table_margin = 40
resolution = (1000, 500)
ball_radius = 15
# physics
# if the velocity of the ball is less then
# friction threshold then it is stopped
friction_threshold = 0.06
friction_coeff = 0.99
# 1 - perfectly elastic ball collisions
# 0 - perfectly inelastic collisions
ball_coeff_of_restitution = 0.9
table_coeff_of_restitution = 0.9

def setup_table():
    """ Sets up the pool table excluding balls

    Returns:
    --------
    tuple
        list
            A list of table_sprites.TableSide instances representing the 
            edges of the table
        numpy.array
            An array containing the positions of holes on the board
            axis 0: hole index
            axis 1: coordinates
    """
    table_sides = []
    table_side_points = np.empty((1, 2))
    # holes_x and holes_y holds the possible xs and ys of the table holes
    # with a position ID in the second tuple field
    # so the top left hole has id 1,1
    holes_x = [(table_margin, 1), (resolution[0] /
                                            2, 2), (resolution[0] - table_margin, 3)]
    holes_y = [(table_margin, 1),
                (resolution[1] - table_margin, 2)]
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
            offset = config.middle_hole_offset
        else:
            offset = config.side_hole_offset
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
            table_sides.append(table_sprites.TableSide(
                [point, table_side_points[num + 1]]))
    table_sides.append(table_sprites.TableSide(
        [table_side_points[-1], table_side_points[0]]))
    return table_sides, all_hole_positions[:, :, 0]