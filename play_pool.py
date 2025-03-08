from poolsim.pool import ball
from poolsim.pool import collisions
from poolsim.pool import config
from poolsim.pool import cue
from poolsim.pool import event
from poolsim.pool import gamestate
from poolsim.pool import graphics
from poolsim.pool import physics

import numpy as np
import pandas as pd

def simulate():
    balls = np.array([ball.Ball() for k in range(5)])
    

if __name__=="__main__":
    simulate()





