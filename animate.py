"""The animation code skeleton was made using chatgpt"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.lines as mlines
from poolsim.pool import config
from play_pool import simulate_and_save_states, simulate

_width = config.resolution[0]
_height = config.resolution[1]

def animate(holes, lines, states, pocketed):
    """Animates a pool shot
    
    Parameters:
    -----------
    holes : numpy.array(shape=(num_holes, 2), dtype=np.int8)
        The positions of the holes (indexed (hole, x/y))
    lines : numpy.array(shape=(n_lines, 2, 2))
        The lines delimiting the table (indexed (line, start/end, x/y)) 
    states : numpy.array(shape=(episode_length, num_balls, 2), dtype=np.float32)
            Axis 0 is the time index
            Axis 1 is the ball index. index = 0 denotes the white ball
            Axis 2 contains the final ball coordinates (x, y)
    pocketed : numpy.array(shape=(episode_length, num_balls), dtype=np.int8)
        For each time, an array with values 1 at index i if corresponding ball is pocketed
    
    Returns:
    --------
    None
    """
    # Plotting Setup
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, _width)
    ax.set_ylim(0, _height)
    ax.set_aspect("equal")

    ax.set_facecolor("green")
    # Draw table edges and pockets
    for line in lines:
        ax.add_line(mlines.Line2D([line[0, 0], line[1, 0]], [line[0, 1], line[1, 1]], color="black", lw=3))

    # Draw holes
    hole_circles = []
    for hole in holes:
        circle = plt.Circle(hole, config.hole_radius, color='black', fill=True, lw=3)
        ax.add_patch(circle)
        hole_circles.append(circle)

    # Create a list for ball patches (will be updated per frame)
    balls = []
    colors = ["red", "blue", "yellow", "purple", "orange", "pink", "cyan"]
    for i in range(states.shape[1]): 
        ball_color = "white" if i == 0 else colors[i % len(colors)]
        ball = plt.Circle((0, 0), config.ball_radius, color=ball_color, fill=True)
        ax.add_patch(ball)
        balls.append(ball)

    # Function to update the ball positions for each frame
    def update(frame):
        # Get current ball positions from the states array
        current_state = states[frame, :, :]  # Shape: (num_balls, 2) -> [ball_index, x_pos, y_pos]
        # Update ball positionss
        for i, ball in enumerate(balls):
            ball.set_center((current_state[i, 0], current_state[i, 1]))
            ball.set_visible(not pocketed[frame, i])
        return balls

    # Create the animation
    ani = FuncAnimation(fig, update, frames=states.shape[0], interval=1000 / 60, blit=False)

    # Show the animation
    plt.show()

if __name__=="__main__":
    state = np.array([[500,
                        200], [550,200],
                        [550,205]], dtype=np.float32)
    pocketed = np.zeros(3, np.int8)
    action = np.array([1.0, 0])
    states, pocketed_over_time, holes, lines = simulate_and_save_states(state, pocketed, action)
    animate(holes, lines, states, pocketed_over_time)