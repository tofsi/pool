"""The animation code skeleton was made using chatgpt"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.lines as mlines
from poolsim.pool import config
from play_pool import simulate_and_save_states

_width = config.resolution[0]
_height = config.resolution[1]

def animate(hole_positions, lines, states):
    # Plotting Setup
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, _width)
    ax.set_ylim(0, _height)

    # Draw table edges and pockets
    for line in lines:
        ax.add_line(mlines.Line2D([line[0, 0], line[1, 0]], [line[0, 1], line[1, 1]], color="black", lw=3))

    # Draw holes
    hole_circles = []
    for hole in hole_positions:
        circle = plt.Circle(hole, config.hole_radius, color='black', fill=False, lw=3)
        ax.add_patch(circle)
        hole_circles.append(circle)

    # Create a list for ball patches (will be updated per frame)
    balls = []
    for _ in range(2):  # Assuming 16 balls, including the white ball
        ball = plt.Circle((0, 0), config.ball_radius, color="black", fill=True)
        ax.add_patch(ball)
        balls.append(ball)

    # Function to update the ball positions for each frame
    def update(frame):
        # Get current ball positions from the states array
        current_state = states[frame, :, :]  # Shape: (num_balls, 2) -> [ball_index, x_pos, y_pos]

        # Update ball positions
        for i, ball in enumerate(balls):
            ball.set_center((current_state[i, 0], current_state[i, 1]))

        return balls

    # Create the animation
    ani = FuncAnimation(fig, update, frames=states.shape[0], interval=1000 / 120, blit=False)

    # Show the animation
    plt.show()

if __name__=="__main__":
    state = np.array([[500,
                        200], [550,200]], dtype=np.float32)
    pocketed = np.zeros(2, np.int8)
    action = np.array([1.0, 0])
    states, pocketed_over_time, holes, lines = simulate_and_save_states(state, pocketed, action)
    print(states)
    print(pocketed_over_time)
    #print(lines)
    #print(holes)
    #print(states.shape)
    animate(holes, lines, states)