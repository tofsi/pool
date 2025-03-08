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
from itertools import combinations, product
_max_force = 15

def get_initial_state(ball_count):
    initial_ball_positions = np.empty((ball_count,2))
    return initial_ball_positions

def action_to_velocity(action):
    force = action[0]
    angle = action[1]
    x = force * _max_force * np.cos(angle * 2 * np.pi)
    y = force * _max_force * np.sin(angle * 2 * np.pi)
    return np.array((x, y))

def setup_table():
    table_sides = []
    table_side_points = np.empty((1, 2))
    # holes_x and holes_y holds the possible xs and ys of the table holes
    # with a position ID in the second tuple field
    # so the top left hole has id 1,1
    holes_x = [(config.table_margin, 1), (config.resolution[0] /
                                            2, 2), (config.resolution[0] - config.table_margin, 3)]
    holes_y = [(config.table_margin, 1),
                (config.resolution[1] - config.table_margin, 2)]
    # next three lines are a hack to make and arrange the hole coordinates
    # in the correct sequence
    all_hole_positions = np.array(
        list(product(holes_y, holes_x)))
    all_hole_positions = np.fliplr(all_hole_positions)
    all_hole_positions = np.vstack(
        (all_hole_positions[:3], np.flipud(all_hole_positions[3:])))
    print(all_hole_positions)
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

def simulate(state, pocketed, action):
    n_balls = state.shape[0]
    # Should return next_state, next_pocketed, n_collisions
    pocketed = pocketed.copy()
    num_collisions = 0
    # Set initial positions of balls
    balls = np.array([ball.Ball() for k in range(n_balls)])
    for i, b in enumerate(balls):
        b.pos = state[i, :]
    
    balls[0].velocity = action_to_velocity(action) # Velocity of white ball
    table_sides, holes = setup_table()
    all_stationary = False
    active_balls = {i : b for i, b in enumerate(balls) if not pocketed[i]}
    iter_count = 0
    while not all_stationary:
        iter_count+=1
        # Check pocketed
        for i, b in active_balls.items():
            for hole in holes:
                if physics.distance_less_equal(b.pos, hole, config.hole_radius):
                    pocketed[i] = 1
                    break
        active_balls = {i : b for i, b in enumerate(balls) if not pocketed[i]}

        # Move balls
        for b in balls:
            b.update()
            # Apply collisions between balls, walls
            for line in table_sides:
                if physics.line_ball_collision_check(line, b):
                    physics.collide_line_ball(line, b)
                    num_collisions += 1
                    break
        


        # Apply collisions between balls
        for b, other_b in combinations(balls, 2):
            if physics.ball_collision_check(b, other_b):
                physics.collide_balls(b, other_b)
                num_collisions += 1 # Add for all collisions

        # print([b.pos for b in active_balls.values()])

        if all([(b.velocity == np.zeros(2, np.float32)).all() for b in active_balls.values()]):
            all_stationary = True
    new_state = np.array([b.pos for b in balls])
    # print(config.resolution)
    return new_state, pocketed, num_collisions
    


        
        
    
    

if __name__=="__main__":
    state = 30*np.array([[config.table_margin + config.resolution[0] // 3,
                        config.table_margin + config.resolution[1] // 3],
                        [config.table_margin + config.resolution[0] // 2,
                        config.table_margin + config.resolution[1] // 2]])
    pocketed = np.zeros(5, np.int8)
    action = np.array([0.0, 1.0])
    simulate(state, pocketed, action)





